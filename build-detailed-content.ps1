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

$sourceCards = foreach ($file in $sourceFiles) {
  $relative = RelativeFrom $sourceRoot $file.FullName
  $content = HtmlEncode (ReadTextFile $file.FullName)
  $size = FormatSize $file.Length
@"
        <article class="source-file-card">
          <h3>$([System.Net.WebUtility]::HtmlEncode($relative))</h3>
          <p class="detail-meta">source file · $size</p>
          <details>
            <summary>Open full source</summary>
            <pre class="detail-code"><code>$content</code></pre>
          </details>
        </article>
"@
}

$extensionCards = foreach ($item in $extensionFiles) {
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

  if ($ext -in @(".md", ".txt")) {
    $body = '<details open><summary>Open full text</summary><pre class="detail-code"><code>' + (HtmlEncode (ReadTextFile $file.FullName)) + '</code></pre></details>'
  } elseif ($ext -eq ".html") {
    $body = '<p>This interactive HTML extension has been copied into the site assets. Preview it below or open it separately.</p><p><a class="open-resource" href="' + $assetUrl + '" target="_blank" rel="noopener">Open original HTML resource</a></p><iframe class="extension-frame" src="' + $assetUrl + '" title="' + (HtmlEncode $relative) + '"></iframe>'
  } elseif ($ext -in @(".png", ".jpg", ".jpeg", ".webp", ".gif")) {
    $body = '<p>This diagram or screenshot resource has been copied into the site assets.</p><img class="extension-image" src="' + $assetUrl + '" alt="' + (HtmlEncode $relative) + '">'
  } else {
    $body = '<p><a class="open-resource" href="' + $assetUrl + '" target="_blank" rel="noopener">Open extension resource</a></p>'
  }

@"
        <article class="extension-file-card">
          <h3>$([System.Net.WebUtility]::HtmlEncode("$day/$relative"))</h3>
          <p class="detail-meta">extension file · $size · $ext</p>
          $body
        </article>
"@
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
  <title>EduRAG Full Code and Extension Library</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header">
    <div class="brand-block">
      <p class="eyebrow">Deep Reference</p>
      <h1>EduRAG Full Code and Extension Library</h1>
      <p class="lead">This page fills in the complete project source files and every day02-day07 extension resource. Generated at: $generatedAt</p>
    </div>
    <nav class="top-tabs" aria-label="详细资料导航">
      <a class="tab-link" href="index.html">Back to learning path</a>
      <a class="tab-link" href="#full-code-library">Full Code Library</a>
      <a class="tab-link" href="#extension-library">Extension Library</a>
    </nav>
  </header>

  <main class="detail-shell">
    <section class="detail-summary">
      <article>
        <strong>$sourceCount</strong>
        <span>source/config/page text files</span>
      </article>
      <article>
        <strong>$extensionCount</strong>
        <span>extension resources</span>
      </article>
      <article>
        <strong>day02-day07</strong>
        <span>daily extension content cataloged</span>
      </article>
    </section>

    <section id="full-code-library" class="detail-section">
      <h2>Full Code Library</h2>
      <p class="muted">The files below come from the completed <code>integrated_qa_system</code> source tree and are sorted by path. Open any item to read the full file content.</p>
      <div class="detail-grid">
$($sourceCards -join "`n")
      </div>
    </section>

    <section id="extension-library" class="detail-section">
      <h2>Extension Library</h2>
      <p class="muted">The resources below come from day02-day07 extension folders. Markdown/TXT files show full text. HTML animations and image diagrams are copied into the site assets and previewed here.</p>
      <div class="detail-grid">
$($extensionCards -join "`n")
      </div>
    </section>
  </main>

  <script src="script.js"></script>
</body>
</html>
"@

Set-Content -Path (Join-Path $siteRoot "details.html") -Value $html -Encoding UTF8
Write-Host "Generated details.html with $sourceCount source files and $extensionCount extension files."
