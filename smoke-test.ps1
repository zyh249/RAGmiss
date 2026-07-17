$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$index = Join-Path $root "index.html"
$css = Join-Path $root "styles.css"
$js = Join-Path $root "script.js"

foreach ($file in @($index, $css, $js)) {
  if (-not (Test-Path $file)) {
    throw "Missing required file: $file"
  }
}

$html = Get-Content -Raw -Encoding UTF8 $index
$styles = Get-Content -Raw -Encoding UTF8 $css
$script = Get-Content -Raw -Encoding UTF8 $js

$requiredHtml = @(
  "EduRAG",
  'id="day02"',
  'id="day03"',
  'id="day04"',
  'id="day05"',
  'id="day06"',
  'id="day07"',
  "FastAPI / WebSocket",
  "Milvus Hybrid Search",
  "WeightedRanker",
  "QueryClassifier",
  "RAGSystem"
)

foreach ($pattern in $requiredHtml) {
  if ($html -notlike "*$pattern*") {
    throw "index.html missing pattern: $pattern"
  }
}

$requiredCss = @(".app-shell", ".day-card", ".flow-map", ".module-card", ".glossary-card", "@media")
foreach ($pattern in $requiredCss) {
  if ($styles -notlike "*$pattern*") {
    throw "styles.css missing pattern: $pattern"
  }
}

$requiredJs = @("activateTab", "localStorage", "navigator.clipboard", "scrollIntoView")
foreach ($pattern in $requiredJs) {
  if ($script -notlike "*$pattern*") {
    throw "script.js missing pattern: $pattern"
  }
}

$externalRefs = @("cdn.jsdelivr", "cdnjs.cloudflare", "unpkg.com", "https://")
foreach ($pattern in $externalRefs) {
  if ($html -like "*$pattern*") {
    throw "index.html contains external dependency: $pattern"
  }
}

Write-Host "Smoke test passed: static EduRAG learning site files look complete."
