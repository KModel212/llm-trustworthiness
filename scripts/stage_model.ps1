param(
    [Parameter(Mandatory = $true)]
    [string]$SourceSnapshot,

    [Parameter(Mandatory = $true)]
    [string]$ModelName
)

$ErrorActionPreference = "Stop"

$source = (Resolve-Path -LiteralPath $SourceSnapshot).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$target = Join-Path $repoRoot (Join-Path "models" $ModelName)

if (-not (Test-Path -LiteralPath (Join-Path $source "config.json"))) {
    throw "Source snapshot is missing config.json: $source"
}

if (Test-Path -LiteralPath $target) {
    $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
    $modelsRoot = Join-Path $repoRoot "models"
    if (-not $resolvedTarget.StartsWith($modelsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clear target outside models/: $resolvedTarget"
    }
    Remove-Item -LiteralPath $target -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $target | Out-Null

function Copy-FileBytes {
    param(
        [Parameter(Mandatory = $true)][string]$SourceFile,
        [Parameter(Mandatory = $true)][string]$DestinationFile
    )

    $destDir = Split-Path -Parent $DestinationFile
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    $inputStream = [System.IO.File]::Open(
        $SourceFile,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
    )
    try {
        $outputStream = [System.IO.File]::Open(
            $DestinationFile,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $outputStream.Dispose()
        }
    } finally {
        $inputStream.Dispose()
    }
}

Get-ChildItem -LiteralPath $source -Force -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($source.Length).TrimStart("\", "/")
    $dest = Join-Path $target $relative
    Copy-FileBytes -SourceFile $_.FullName -DestinationFile $dest
}

$required = @("config.json", "tokenizer_config.json")
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $target $file))) {
        throw "Staged model is missing required file: $file"
    }
}

$hasWeights = @(
    Get-ChildItem -LiteralPath $target -Force -File |
        Where-Object { $_.Extension -in @(".safetensors", ".bin") }
).Count -gt 0
if (-not $hasWeights) {
    throw "Staged model has no .safetensors or .bin weights: $target"
}

$reparseFiles = @(
    Get-ChildItem -LiteralPath $target -Force -Recurse |
        Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }
)
if ($reparseFiles.Count -gt 0) {
    throw "Staged model contains reparse-point files; restage outside OneDrive or disable placeholders."
}

Write-Output "Staged model snapshot:"
Write-Output "  Source: $source"
Write-Output "  Target: $target"
Get-ChildItem -LiteralPath $target -Force | Select-Object Name,Length,Mode,LinkType
