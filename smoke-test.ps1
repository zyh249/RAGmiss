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
  'id="day08"',
  'id="day09"',
  "FastAPI / WebSocket",
  "Milvus Hybrid Search",
  "WeightedRanker",
  "QueryClassifier",
  "RAGSystem",
  "RAGAS",
  "Docker",
  "EduRAG 集成问答系统学习站"
)

$requiredDetails = @(
  "EduRAG 源码全文与扩展资料库",
  "day02 到 day09",
  "返回学习路线",
  "资料目录",
  'id="detailReader"',
  'id="tocToggle"',
  "收起目录",
  'id="detailSearch"',
  "toc-list",
  "day-divider",
  "源码全文",
  "扩展资料库",
  "单独打开演示页面",
  "extension-preview",
  "preview-details",
  'loading="lazy"',
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

$removedEnglishDetails = @("Full Code Library", "Extension Library", "Back to learning path", "Open original HTML resource")
foreach ($pattern in $removedEnglishDetails) {
  if ($detailHtml -like "*$pattern*") {
    throw "details.html still contains old English label: $pattern"
  }
}

$sourceFileCount = ([regex]::Matches($detailHtml, 'class="source-file-card"')).Count
if ($sourceFileCount -lt 20) {
  throw "details.html source file count too low: $sourceFileCount"
}

$extensionFileCount = ([regex]::Matches($detailHtml, 'class="extension-file-card"')).Count
if ($extensionFileCount -lt 120) {
  throw "details.html extension file count too low: $extensionFileCount"
}

$requiredCss = @(".app-shell", ".day-card", ".flow-map", ".module-card", ".glossary-card", ".detail-page", ".detail-reader", ".toc-collapsed", ".toc-toggle", ".detail-toc", ".toc-list", ".extension-intro", ".extension-preview", "@media")
foreach ($pattern in $requiredCss) {
  if ($styles -notlike "*$pattern*") {
    throw "styles.css missing pattern: $pattern"
  }
}

$requiredJs = @("activateTab", "localStorage", "navigator.clipboard", "scrollIntoView", "detailSearch", "data-detail-item", "setTocCollapsed", "edurag-detail-toc-collapsed", "tocToggle")
foreach ($pattern in $requiredJs) {
  if ($script -notlike "*$pattern*") {
    throw "script.js missing pattern: $pattern"
  }
}

$encodingArtifacts = @("澶嶅埗", "宸插", "鍗曠嫭", "闆嗘垚", "瀛︿範", "鎵╁睍")
foreach ($pattern in $encodingArtifacts) {
  if ($html -like "*$pattern*" -or $script -like "*$pattern*" -or $detailHtml -like "*$pattern*") {
    throw "Generated files contain encoding artifact: $pattern"
  }
}

$externalRefs = @("cdn.jsdelivr", "cdnjs.cloudflare", "unpkg.com", "https://")
foreach ($pattern in $externalRefs) {
  if ($html -like "*$pattern*") {
    throw "index.html contains external dependency: $pattern"
  }
}

Write-Host "Smoke test passed: static EduRAG learning site files look complete."




