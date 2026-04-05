param(
    [string]$State = "docs/Literature/asm_volume9_loop_state.json",
    [string]$TmpDir = "tmp/asm_book_loop",
    [string]$WindowTitle = "Codex",
    [int]$PollSeconds = 15,
    [string]$LogFile = "tmp/asm_book_loop/runner.log",
    [switch]$AllowForegroundFallback
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RepoRoot {
    if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        throw "Could not resolve script root."
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-LoopStateJson([string]$statePath) {
    return & python -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); print(json.dumps(d, ensure_ascii=False))" $statePath
}

function Get-LoopSnapshot([string]$statePath) {
    $json = Get-LoopStateJson $statePath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) {
        throw "Failed to read loop state."
    }
    return $json | ConvertFrom-Json
}

function Find-CodexWindow([string]$windowTitle) {
    Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $windows = $root.FindAll(
        [System.Windows.Automation.TreeScope]::Children,
        (New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Window
        ))
    )
    for ($i = 0; $i -lt $windows.Count; $i++) {
        $window = $windows.Item($i)
        if (([string]$window.Current.Name) -like "*$windowTitle*") {
            return $window
        }
    }
    return $null
}

function Test-CompletionMarkerSeen([string]$windowTitle, [string]$marker) {
    if ([string]::IsNullOrWhiteSpace($marker)) {
        return $false
    }
    $window = Find-CodexWindow $windowTitle
    if ($null -eq $window) {
        return $false
    }
    $all = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
    for ($i = 0; $i -lt $all.Count; $i++) {
        $el = $all.Item($i)
        $name = [string]$el.Current.Name
        if (-not [string]::IsNullOrWhiteSpace($name) -and $name.Contains($marker)) {
            return $true
        }
    }
    return $false
}

function Write-RunnerLog([string]$logPath, [string]$message) {
    $dir = Split-Path -Parent $logPath
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force $dir | Out-Null
    }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$stamp] $message"
}

function Invoke-Sender(
    [string]$repoRoot,
    [string]$statePath,
    [string]$tmpPath,
    [string]$windowTitle,
    [switch]$generateNext,
    [switch]$allowForegroundFallback
) {
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts/send_codex_pending.ps1"),
        "-State", $statePath,
        "-TmpDir", $tmpPath,
        "-WindowTitle", $windowTitle
    )
    if ($generateNext) {
        $args += "-GenerateNext"
    }
    if ($allowForegroundFallback) {
        $args += "-AllowForegroundFallback"
    }
    & powershell @args
    return $LASTEXITCODE
}

$repoRoot = Resolve-RepoRoot
$statePath = if ([System.IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $repoRoot $State }
$tmpPath = if ([System.IO.Path]::IsPathRooted($TmpDir)) { $TmpDir } else { Join-Path $repoRoot $TmpDir }
$logPath = if ([System.IO.Path]::IsPathRooted($LogFile)) { $LogFile } else { Join-Path $repoRoot $LogFile }

$lastSentKey = ""
Write-RunnerLog $logPath "Runner started."

while ($true) {
    try {
        $snapshot = Get-LoopSnapshot $statePath
        $completed = [int]$snapshot.completed_through_page
        $finalPage = [int]$snapshot.book_end_page
        $pending = $snapshot.pending_batch

        if ($completed -ge $finalPage -and $null -eq $pending) {
            Write-RunnerLog $logPath "Book completed through page $completed. Runner stopped."
            break
        }

        if ($null -eq $pending) {
            Write-RunnerLog $logPath "No pending batch. Generating and sending next prompt."
            $exitCode = Invoke-Sender -repoRoot $repoRoot -statePath $statePath -tmpPath $tmpPath -windowTitle $WindowTitle -generateNext -allowForegroundFallback:$AllowForegroundFallback
            if ($exitCode -eq 0) {
                $snapshot = Get-LoopSnapshot $statePath
                if ($null -ne $snapshot.pending_batch) {
                    $lastSentKey = [string]$snapshot.pending_batch.prompt_file + "|" + [string]$snapshot.pending_batch.generated_at
                }
                Write-RunnerLog $logPath "Generated and sent next prompt."
            }
            else {
                Write-RunnerLog $logPath "Generate/send failed. Will retry."
            }
        }
        else {
            $currentKey = [string]$pending.prompt_file + "|" + [string]$pending.generated_at
            $alreadySent = $null -ne $pending.sent_at -and [string]$pending.sent_at -ne ""
            if ($alreadySent -and (Test-CompletionMarkerSeen -windowTitle $WindowTitle -marker ([string]$pending.completion_marker))) {
                Write-RunnerLog $logPath "Completion marker detected for pending batch. Marking complete."
                & python (Join-Path $repoRoot "scripts/asm_book_loop.py") --state $statePath complete
                if ($LASTEXITCODE -eq 0) {
                    Write-RunnerLog $logPath "Pending batch marked complete."
                    $lastSentKey = ""
                }
                else {
                    Write-RunnerLog $logPath "Failed to mark pending batch complete."
                }
            }
            elseif ($alreadySent) {
                $lastSentKey = $currentKey
            }
            elseif ($currentKey -ne $lastSentKey) {
                Write-RunnerLog $logPath "Sending pending prompt $currentKey."
                $exitCode = Invoke-Sender -repoRoot $repoRoot -statePath $statePath -tmpPath $tmpPath -windowTitle $WindowTitle -allowForegroundFallback:$AllowForegroundFallback
                if ($exitCode -eq 0) {
                    $lastSentKey = $currentKey
                    Write-RunnerLog $logPath "Pending prompt sent."
                }
                else {
                    Write-RunnerLog $logPath "Pending prompt send failed. Will retry."
                }
            }
        }
    }
    catch {
        Write-RunnerLog $logPath ("Runner error: " + $_.Exception.Message)
    }
    Start-Sleep -Seconds ([Math]::Max(3, $PollSeconds))
}
