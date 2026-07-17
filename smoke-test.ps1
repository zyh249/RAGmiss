$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$index = Join-Path $root "index.html"
$details = Join-Path $root "details.html"
$css = Join-Path $root "styles.css"
$js = Join-Path $root "script.js"

foreach ($file in @($index, $details, $css, $js)) {
  if (-not (Test-Path $file)) {
    throw "Missing required file: $file"
  }
}

$html = Get-Content -Raw -Encoding UTF8 $index
$detailHtml = Get-Content -Raw -Encoding UTF8 $details
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

$requiredDetails = @(
  "Full Code Library",
  "Extension Library",
  "app.py",
  "new_main.py",
  "mysql_qa/retrieval/bm25_search.py",
  "rag_qa/core/vector_store.py",
  "rag_qa/core/new_rag_system.py",
  "bm25_code_visualizer.html",
  "vectorstore_logic_animation.html",
  "BERT_QueryClassifier",
  "hybrid_search_rerank_animation.html"
)

foreach ($pattern in $requiredHtml) {
  if ($html -notlike "*$pattern*") {
    throw "index.html missing pattern: $pattern"
  }
}

foreach ($pattern in $requiredDetails) {
  if ($detailHtml -notlike "*$pattern*") {
    throw "details.html missing pattern: $pattern"
  }
}

$sourceFileCount = ([regex]::Matches($detailHtml, 'class="source-file-card"')).Count
if ($sourceFileCount -lt 20) {
  throw "details.html source file count too low: $sourceFileCount"
}

$extensionFileCount = ([regex]::Matches($detailHtml, 'class="extension-file-card"')).Count
if ($extensionFileCount -lt 40) {
  throw "details.html extension file count too low: $extensionFileCount"
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
