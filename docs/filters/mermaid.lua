--[[
  Pandoc Lua filter to render Mermaid diagrams to SVG images.
  
  Usage:
    pandoc input.md --lua-filter=docs/filters/mermaid.lua -o output.pdf
  
  Requirements:
    - mermaid-cli (mmdc): npm install -g @mermaid-js/mermaid-cli
    
  Behavior:
    - Detects fenced code blocks with class "mermaid"
    - Computes SHA256 hash of diagram content for caching
    - Renders to SVG using mmdc if not already cached
    - Replaces code block with image reference
]]

local debug_mode = os.getenv("MERMAID_DEBUG") == "1"

-- Directories are relative to the working directory where pandoc runs
-- build_docs.sh runs pandoc from docs/, so these are relative to docs/
local DIAGRAMS_DIR = "diagrams"
local CACHE_DIR = "_cache"

-- Simple SHA256 implementation using external command
-- Falls back to MD5 if sha256sum not available
local function compute_hash(content)
  -- Write content to temp file to avoid shell escaping issues
  local temp_file = os.tmpname()
  local f = io.open(temp_file, "w")
  if not f then
    error("Failed to create temp file for hashing")
  end
  f:write(content)
  f:close()
  
  -- Try sha256sum first (Linux), then shasum (macOS), then md5
  local hash = nil
  local commands = {
    "sha256sum " .. temp_file .. " 2>/dev/null",
    "shasum -a 256 " .. temp_file .. " 2>/dev/null",
    "md5 -q " .. temp_file .. " 2>/dev/null",
    "md5sum " .. temp_file .. " 2>/dev/null"
  }
  
  for _, cmd in ipairs(commands) do
    local handle = io.popen(cmd)
    if handle then
      local result = handle:read("*a")
      handle:close()
      if result and result ~= "" then
        -- Extract the hash (first word)
        hash = result:match("^(%w+)")
        if hash then
          break
        end
      end
    end
  end
  
  os.remove(temp_file)
  
  if not hash then
    -- Fallback: simple string hash
    local h = 0
    for i = 1, #content do
      h = (h * 31 + content:byte(i)) % 2147483647
    end
    hash = string.format("%08x", h)
  end
  
  -- Truncate to reasonable length
  return hash:sub(1, 16)
end

-- Check if a file exists
local function file_exists(path)
  local f = io.open(path, "r")
  if f then
    f:close()
    return true
  end
  return false
end

-- Create directory if it doesn't exist (cross-platform)
local function ensure_dir(path)
  -- Use mkdir -p on Unix, works on both Linux and macOS
  local cmd
  if package.config:sub(1,1) == '\\' then
    -- Windows
    cmd = 'if not exist "' .. path:gsub("/", "\\") .. '" mkdir "' .. path:gsub("/", "\\") .. '"'
  else
    -- Unix (Linux, macOS)
    cmd = 'mkdir -p "' .. path .. '"'
  end
  os.execute(cmd)
end

-- Log message if debug mode is enabled
local function log(msg)
  if debug_mode then
    io.stderr:write("[mermaid.lua] " .. msg .. "\n")
  end
end

-- Render Mermaid diagram to PNG (PNG embeds fonts properly in PDFs)
local function render_mermaid(content, output_path)
  local mmd_path = output_path:gsub("%.png$", ".mmd")
  
  -- Write Mermaid source to cache
  local f = io.open(mmd_path, "w")
  if not f then
    error("Failed to write Mermaid source to: " .. mmd_path)
  end
  f:write(content)
  f:close()
  
  -- Render with mmdc - use PNG with high scale for quality
  local cmd = string.format(
    'mmdc -i "%s" -o "%s" -b white -s 2 --quiet 2>&1',
    mmd_path, output_path
  )
  
  log("Running: " .. cmd)
  
  local handle = io.popen(cmd)
  local result = handle:read("*a")
  local success = handle:close()
  
  if not success or not file_exists(output_path) then
    error("Mermaid rendering failed for " .. mmd_path .. "\nOutput: " .. (result or "no output"))
  end
  
  log("Generated: " .. output_path)
  return true
end

-- Main CodeBlock filter
function CodeBlock(el)
  -- Check if this is a Mermaid block
  local is_mermaid = false
  for _, class in ipairs(el.classes) do
    if class == "mermaid" then
      is_mermaid = true
      break
    end
  end
  
  if not is_mermaid then
    return nil
  end
  
  -- Compute hash for caching
  local hash = compute_hash(el.text)
  local filename = "mermaid-" .. hash .. ".png"
  local output_path = DIAGRAMS_DIR .. "/" .. filename
  local cache_mmd_path = CACHE_DIR .. "/mermaid-" .. hash .. ".mmd"
  
  log("Processing Mermaid block, hash: " .. hash)
  
  -- Ensure directories exist
  ensure_dir(DIAGRAMS_DIR)
  ensure_dir(CACHE_DIR)
  
  -- Check if already rendered (cache hit)
  if file_exists(output_path) then
    log("Cache hit: " .. output_path)
  else
    log("Cache miss, rendering: " .. output_path)
    -- Write to cache dir and render
    local f = io.open(cache_mmd_path, "w")
    if f then
      f:write(el.text)
      f:close()
    end
    
    -- Render the diagram
    local success, err = pcall(function()
      render_mermaid(el.text, output_path)
    end)
    
    if not success then
      io.stderr:write("ERROR: " .. tostring(err) .. "\n")
      -- Return original block on error so user sees the source
      return el
    end
  end
  
  -- Return an image element pointing to the rendered PNG
  -- Use relative path from the document location
  return pandoc.Para({
    pandoc.Image(
      {},                    -- alt text (empty)
      output_path,           -- image path
      ""                     -- title
    )
  })
end

-- Return the filter
return {
  { CodeBlock = CodeBlock }
}
