-- WRK Lua script for POST requests to search endpoint

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

local queries = {
  '{"query": "French noir films", "limit": 10}',
  '{"query": "documentaries about climate", "limit": 5}',
  '{"query": "romantic comedies", "limit": 10}',
  '{"query": "action thriller", "limit": 10}',
  '{"query": "philosophical drama", "limit": 10}'
}

local counter = 0

request = function()
  counter = counter + 1
  local body = queries[(counter % #queries) + 1]
  return wrk.format("POST", nil, nil, body)
end

done = function(summary, latency, requests)
  io.write("----------------------------------------\n")
  io.write("Requests: " .. summary.requests .. "\n")
  io.write("Duration: " .. summary.duration / 1000000 .. "s\n")
  io.write("QPS: " .. string.format("%.2f", summary.requests / (summary.duration / 1000000)) .. "\n")
  io.write("Avg Latency: " .. string.format("%.2f", latency.mean / 1000) .. "ms\n")
  io.write("p50 Latency: " .. string.format("%.2f", latency:percentile(50) / 1000) .. "ms\n")
  io.write("p99 Latency: " .. string.format("%.2f", latency:percentile(99) / 1000) .. "ms\n")
  io.write("----------------------------------------\n")
end
