# Download and extract Vosk Russian small model for AnswerBot
# Model: vosk-model-small-ru-0.22 (~45 MB)
# Place in: app/src/main/assets/answerbot/vosk-model-small-ru-0.22/

$ErrorActionPreference = 'Stop'
$modelUrl = 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'
$workDir = "$PSScriptRoot\..\app\src\main\assets\answerbot"
$zipPath = "$env:TEMP\vosk-model-small-ru-0.22.zip"
$modelDir = "$workDir\vosk-model-small-ru-0.22"

if (Test-Path -LiteralPath $modelDir) {
    Write-Host "Model already exists at $modelDir"
    $files = Get-ChildItem -LiteralPath $modelDir -Recurse | Measure-Object
    Write-Host "$($files.Count) files present"
    exit 0
}

Write-Host "Downloading Vosk model ($modelUrl)..."
try {
    Invoke-WebRequest -Uri $modelUrl -OutFile $zipPath -UseBasicParsing
} catch {
    Write-Error "Download failed: $_"
    Write-Host "Manual download: $modelUrl"
    Write-Host "Extract to: $modelDir"
    exit 2
}

Write-Host "Extracting..."
try {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $workDir -Force
} catch {
    Write-Error "Extract failed: $_"
    exit 2
}

Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $modelDir) {
    $files = Get-ChildItem -LiteralPath $modelDir -Recurse | Measure-Object
    Write-Host "Done. $($files.Count) files in $modelDir"
} else {
    Write-Error "Model directory not found after extract. Expected: $modelDir"
    exit 2
}
