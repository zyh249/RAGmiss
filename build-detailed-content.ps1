$ErrorActionPreference = "Stop"

$siteRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $siteRoot
$sourceRoot = Get-ChildItem -Path $projectRoot -Directory -Filter "integrated_qa_system" -Recurse |
  Where-Object { Test-Path (Join-Path $_.FullName "app.py") } |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $sourceRoot) {
  throw "Cannot locate integrated_qa_system source root containing app.py"
}

$extensionRoots = Get-ChildItem -Path $projectRoot -Directory -Filter "day*" |
  Sort-Object Name |
  ForEach-Object {
    $dayName = $_.Name
    $dayKey = if ($dayName -match "day(\d+)") { "day$($Matches[1])" } else { $dayName }
    $extensionDir = Get-ChildItem -Path $_.FullName -Directory -Filter "03-*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($extensionDir) {
      @{ Day = $dayKey; Path = $extensionDir.FullName }
    }
  } |
  Where-Object { $_ -and $_.Day -match "^day0[2-7]$" }

function HtmlEncode([string]$value) {
  return [System.Net.WebUtility]::HtmlEncode($value)
}

function UrlPath([string]$relativePath) {
  $parts = $relativePath -split "[\\/]+"
  return ($parts | ForEach-Object { [System.Uri]::EscapeDataString($_) }) -join "/"
}

function ReadTextFile([string]$path) {
  try {
    return Get-Content -Raw -Encoding UTF8 $path
  } catch {
    try {
      return Get-Content -Raw -Encoding Default $path
    } catch {
      return "[Cannot read text content] $($_.Exception.Message)"
    }
  }
}

function FormatSize([long]$bytes) {
  if ($bytes -ge 1MB) { return "{0:N2} MB" -f ($bytes / 1MB) }
  if ($bytes -ge 1KB) { return "{0:N1} KB" -f ($bytes / 1KB) }
  return "$bytes B"
}

function RelativeFrom([string]$base, [string]$path) {
  $baseFull = [System.IO.Path]::GetFullPath($base).TrimEnd('\') + '\'
  $pathFull = [System.IO.Path]::GetFullPath($path)
  return $pathFull.Substring($baseFull.Length).Replace('\', '/')
}

$codeExtensions = @(".py", ".ini", ".txt", ".md", ".html", ".yml", ".yaml")
$sourceFiles = Get-ChildItem -Recurse -File $sourceRoot |
  Where-Object {
    $codeExtensions -contains $_.Extension.ToLowerInvariant() -and
    $_.FullName -notmatch "\\__pycache__\\" -and
    $_.FullName -notmatch "\\logs\\"
  } |
  Sort-Object FullName

$extensionFiles = foreach ($rootInfo in $extensionRoots) {
  if (Test-Path $rootInfo.Path) {
    Get-ChildItem -Recurse -File $rootInfo.Path | ForEach-Object {
      [pscustomobject]@{
        Day = $rootInfo.Day
        Base = $rootInfo.Path
        File = $_
        Relative = RelativeFrom $rootInfo.Path $_.FullName
      }
    }
  }
}

$extensionAssetRoot = Join-Path $siteRoot "assets\extensions"
New-Item -ItemType Directory -Force -Path $extensionAssetRoot | Out-Null

$sourceCards = New-Object System.Collections.Generic.List[string]
$sourceTocLinks = New-Object System.Collections.Generic.List[string]
$sourceIndex = 0

foreach ($file in $sourceFiles) {
  $sourceIndex++
  $relative = RelativeFrom $sourceRoot $file.FullName
  $content = HtmlEncode (ReadTextFile $file.FullName)
  $size = FormatSize $file.Length
  $sourceId = "source-$sourceIndex"
  $encodedRelative = HtmlEncode $relative
  $sourceTocLinks.Add("<a href=""#$sourceId"" data-detail-link=""source"">$encodedRelative</a>") | Out-Null
  $sourceCards.Add(@"
        <article id="$sourceId" class="source-file-card" data-detail-item data-detail-text="$encodedRelative">
          <h3>$([System.Net.WebUtility]::HtmlEncode($relative))</h3>
          <p class="detail-meta">源码文件 · $size</p>
          <details>
            <summary>展开完整源码</summary>
            <pre class="detail-code"><code>$content</code></pre>
          </details>
        </article>
"@) | Out-Null
}

$extensionCards = New-Object System.Collections.Generic.List[string]
$extensionTocLinks = New-Object System.Collections.Generic.List[string]
$extensionDayLinks = New-Object System.Collections.Generic.List[string]
$extensionGroups = @($extensionFiles) | Sort-Object Day, Relative | Group-Object Day
$extensionIndex = 0

foreach ($group in $extensionGroups) {
  $dayId = "ext-$($group.Name)"
  $extensionDayLinks.Add("<a href=""#$dayId"">$($group.Name)<span>$($group.Count)</span></a>") | Out-Null
  $extensionCards.Add(@"
        <div id="$dayId" class="day-divider">
          <strong>$($group.Name)</strong>
          <span>$($group.Count) 个扩展文件</span>
        </div>
"@) | Out-Null

foreach ($item in $group.Group) {
  $extensionIndex++
  $file = $item.File
  $relative = $item.Relative
  $day = $item.Day
  $targetDir = Join-Path $extensionAssetRoot (Join-Path $day (Split-Path -Parent $relative))
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  $targetPath = Join-Path $targetDir $file.Name
  Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
  $assetRelative = "assets/extensions/$day/$relative"
  $assetUrl = UrlPath $assetRelative
  $size = FormatSize $file.Length
  $ext = $file.Extension.ToLowerInvariant()
  $extensionId = "extension-$extensionIndex"
  $extensionTitle = "$day/$relative"
  $encodedExtensionTitle = HtmlEncode $extensionTitle
  $extensionTocLinks.Add("<a href=""#$extensionId"" data-detail-link=""extension"">$encodedExtensionTitle</a>") | Out-Null

  if ($ext -in @(".md", ".txt")) {
    $body = '<details><summary>展开完整文本</summary><pre class="detail-code"><code>' + (HtmlEncode (ReadTextFile $file.FullName)) + '</code></pre></details>'
  } elseif ($ext -eq ".html") {
    $body = '<div class="extension-intro"><p>这是一个可交互的 HTML 扩展演示。页面内预览默认折叠，避免完整资料库一次加载太长；需要沉浸查看时可以单独打开。</p><a class="open-resource" href="' + $assetUrl + '" target="_blank" rel="noopener">单独打开演示页面</a></div><details class="preview-details"><summary>展开页面内预览</summary><div class="extension-preview"><iframe class="extension-frame" src="' + $assetUrl + '" title="' + (HtmlEncode $relative) + '" loading="lazy"></iframe></div></details>'
  } elseif ($ext -in @(".png", ".jpg", ".jpeg", ".webp", ".gif")) {
    $body = '<p>这是扩展资料中的图例或截图，已经复制到网站资源目录。</p><img class="extension-image" src="' + $assetUrl + '" alt="' + (HtmlEncode $relative) + '">'
  } else {
    $body = '<p><a class="open-resource" href="' + $assetUrl + '" target="_blank" rel="noopener">打开扩展资源</a></p>'
  }

  $extensionCards.Add(@"
        <article id="$extensionId" class="extension-file-card" data-detail-item data-detail-text="$encodedExtensionTitle">
          <h3>$([System.Net.WebUtility]::HtmlEncode("$day/$relative"))</h3>
          <p class="detail-meta">扩展文件 · $size · $ext</p>
          $body
        </article>
"@) | Out-Null
}
}

$sourceCount = $sourceFiles.Count
$extensionCount = @($extensionFiles).Count
$generatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$html = @"
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EduRAG 源码全文与扩展资料库</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body class="detail-page">
  <header class="site-header">
    <div class="brand-block">
      <p class="eyebrow">完整参考资料</p>
      <h1>EduRAG 源码全文与扩展资料库</h1>
      <p class="lead">这里集中整理完整项目源码，以及 day02 到 day07 的所有扩展资料、动画演示和图例。生成时间：$generatedAt</p>
    </div>
    <nav class="top-tabs" aria-label="详细资料导航">
      <a class="tab-link" href="index.html">返回学习路线</a>
      <a class="tab-link" href="#detail-directory">资料目录</a>
      <a class="tab-link" href="#full-code-library">源码全文</a>
      <a class="tab-link" href="#extension-library">扩展资料库</a>
    </nav>
  </header>

  <main class="detail-shell">
    <section class="detail-summary">
      <article>
        <strong>$sourceCount</strong>
        <span>源码、配置和页面文件</span>
      </article>
      <article>
        <strong>$extensionCount</strong>
        <span>扩展资料文件</span>
      </article>
      <article>
        <strong>day02-day07</strong>
        <span>每日扩展内容已归档</span>
      </article>
    </section>

    <div id="detailReader" class="detail-reader">
      <div class="reader-toolbar">
        <button id="tocToggle" class="toc-toggle" type="button" aria-controls="detail-directory" aria-expanded="true">收起目录</button>
        <span>目录可以随时展开或收起，正文区域会自动调整宽度。</span>
      </div>
      <aside id="detail-directory" class="detail-toc" aria-label="完整资料库目录">
        <div class="toc-panel">
          <p class="eyebrow">资料目录</p>
          <label class="search-label" for="detailSearch">搜索源码或扩展文件</label>
          <input id="detailSearch" class="glossary-search" type="search" placeholder="输入 app.py、BM25、day06、RAGSystem...">
          <nav class="toc-quick">
            <a href="#full-code-library">源码全文<span>$sourceCount</span></a>
            <a href="#extension-library">扩展资料库<span>$extensionCount</span></a>
          </nav>
          <h3>按天查看扩展</h3>
          <nav class="toc-quick compact">
$($extensionDayLinks -join "`n")
          </nav>
          <details open>
            <summary>源码目录</summary>
            <div class="toc-list">
$($sourceTocLinks -join "`n")
            </div>
          </details>
          <details>
            <summary>扩展目录</summary>
            <div class="toc-list">
$($extensionTocLinks -join "`n")
            </div>
          </details>
        </div>
      </aside>

      <div class="detail-content">
        <section id="full-code-library" class="detail-section">
          <h2>源码全文</h2>
          <p class="muted">下面的文件来自完整项目 <code>integrated_qa_system</code>，按照路径排序。展开任意文件即可阅读完整内容。</p>
          <div class="detail-grid">
$($sourceCards -join "`n")
          </div>
        </section>

        <section id="extension-library" class="detail-section">
          <h2>扩展资料库</h2>
          <p class="muted">下面的资源来自 day02 到 day07 的扩展文件夹。Markdown 和 TXT 默认折叠，HTML 动画提供折叠预览和单独打开入口，避免完整资料库一次展开过长。</p>
          <div class="detail-grid">
$($extensionCards -join "`n")
          </div>
        </section>
      </div>
    </div>
  </main>

  <script src="script.js"></script>
</body>
</html>
"@

Set-Content -Path (Join-Path $siteRoot "details.html") -Value $html -Encoding UTF8
Write-Host "Generated details.html with $sourceCount source files and $extensionCount extension files."



