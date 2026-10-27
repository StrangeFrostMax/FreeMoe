# Record one reproducible FreeMoe correctness/benchmark run on Windows.
# Usage: powershell -File scripts/run_record.ps1 configs/example.env
param(
    [Parameter(Mandatory = $true)]
    [string]$Config
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Config)) {
    Write-Error "config file not found: $Config"
    exit 2
}

$cfg = @{}
Get-Content $Config | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        $cfg[$matches[1].Trim()] = $matches[2].Trim()
    }
}
foreach ($required in @("RUN_ID", "MODEL", "BACKEND")) {
    if (-not $cfg.ContainsKey($required) -or [string]::IsNullOrWhiteSpace($cfg[$required])) {
        Write-Error "missing required config key: $required"
        exit 2
    }
}
function Get-ConfigValue($name, $default) {
    if ($cfg.ContainsKey($name) -and -not [string]::IsNullOrWhiteSpace($cfg[$name])) { return $cfg[$name] }
    return $default
}

$runId = $cfg["RUN_ID"]
$model = $cfg["MODEL"]
$backend = $cfg["BACKEND"]
$reader = Get-ConfigValue "READER" "buffered"
$cacheSlots = Get-ConfigValue "CACHE_SLOTS" "1"
$tokens = Get-ConfigValue "TOKENS" "16"
$limit = Get-ConfigValue "LIMIT" "20"
$prefetch = Get-ConfigValue "PREFETCH" "0"
$referenceRoot = Get-ConfigValue "REFERENCE_ROOT" "tests/reference"
$extra = Get-ConfigValue "EXTRA_ARGS" ""

$out = Join-Path "results" $runId
New-Item -ItemType Directory -Force -Path $out | Out-Null
Copy-Item $Config (Join-Path $out "config.env") -Force
(git rev-parse HEAD) | Out-File (Join-Path $out "commit.txt") -Encoding ascii

$machine = @()
$machine += "# system"
$machine += (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, TotalVisibleMemorySize | Format-List | Out-String)
$machine += "# cpu"
$machine += (Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String)
$machine += "# gpu"
$machine += (Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Format-List | Out-String)
$machine += "# backend"
$machine += ((& .\build\freemoe.exe device-info --backend $backend --json) | Out-String)
$machine | Out-File (Join-Path $out "machine.txt") -Encoding utf8

$freemoeArgs = @(
    "compare", "--model", $model, "--backend", $backend,
    "--reader", $reader, "--cache-slots", $cacheSlots,
    "--tokens", $tokens, "--limit", $limit,
    "--prefetch", $prefetch, "--reference-root", $referenceRoot,
    "--run-id", $runId, "--results-dir", "results"
)
if (-not [string]::IsNullOrWhiteSpace($extra)) {
    $freemoeArgs += ($extra -split '\s+')
}
$stdout = Join-Path $out "stdout.json"
$stderr = Join-Path $out "stderr.txt"
& .\build\freemoe.exe @freemoeArgs 1> $stdout 2> $stderr
$rc = $LASTEXITCODE
$rc | Out-File (Join-Path $out "exit_code.txt") -Encoding ascii
python scripts/update_results_index.py | Out-File (Join-Path $out "results_index.txt") -Encoding utf8
Get-ChildItem $out -File | Where-Object { $_.Name -ne "SHA256SUMS.txt" } | ForEach-Object {
    $hash = Get-FileHash $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
} | Out-File (Join-Path $out "SHA256SUMS.txt") -Encoding ascii
Write-Host "recorded $out (exit=$rc)"
exit $rc
