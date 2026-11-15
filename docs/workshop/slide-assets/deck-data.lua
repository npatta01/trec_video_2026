local directory = pandoc.path.directory(PANDOC_SCRIPT_FILE)

local function read_asset(parts)
  local path = pandoc.path.join(parts)
  local source = assert(io.open(path, "r"), "Missing deck input: " .. path)
  local text = source:read("*a")
  source:close()
  return text
end

local function insert_asset(text, placeholder, parts, format)
  local first, last = text:find(placeholder, 1, true)
  if not first then return text end
  local asset = read_asset(parts)
  return text:sub(1, first - 1) .. (format and format(asset) or asset) .. text:sub(last + 1)
end

function RawBlock(block)
  if block.format ~= "html" then return end

  local text = insert_asset(block.text, "<!-- official-leaderboard-data -->", {
    directory, "..", "..", "experiments", "artifacts",
    "trec-vqa-2026-official-results", "leaderboard-summary.json"
  }, function(data)
    data = data:gsub("<", [[\u003c]])
    return '<script type="application/json" id="official-leaderboard-data">' .. data .. "</script>"
  end)

  text = insert_asset(text, "<!-- private-weather-transcript -->", {
    directory, "..", "..", "paper", "private-frames", "weather-transcript.html"
  })
  text = insert_asset(text, "<!-- private-speech-transcript -->", {
    directory, "..", "..", "paper", "private-frames", "speech-transcript.html"
  })
  if text == block.text then return end
  block.text = text
  return block
end
