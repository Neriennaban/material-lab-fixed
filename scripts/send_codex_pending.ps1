param(
    [string]$State = "docs/Literature/asm_volume9_loop_state.json",
    [string]$TmpDir = "tmp/asm_book_loop",
    [string]$WindowTitle = "Codex",
    [switch]$GenerateNext,
    [switch]$PrintOnly,
    [switch]$AllowForegroundFallback
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RepoRoot {
    $scriptDir = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptDir)) {
        throw "Could not resolve script root."
    }
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Get-PendingPromptPathFromState([string]$statePath) {
    if (-not (Test-Path $statePath)) {
        throw "State file not found: $statePath"
    }
    $promptPath = & python -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); p=d.get('pending_batch') or {}; print(p.get('prompt_file',''))" $statePath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read pending batch from state: $statePath"
    }
    return ([string]$promptPath).Trim()
}

function Get-PendingCompletionMarker([string]$statePath) {
    $marker = & python -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); p=d.get('pending_batch') or {}; print(p.get('completion_marker',''))" $statePath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read completion marker from state: $statePath"
    }
    return ([string]$marker).Trim()
}

function Mark-PendingSentInState([string]$statePath) {
    & python -c "import json,sys,datetime; p=sys.argv[1]; d=json.load(open(p, encoding='utf-8')); pb=d.get('pending_batch'); pb is not None and pb.__setitem__('sent_at', datetime.datetime.now().astimezone().isoformat(timespec='seconds')); json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)" $statePath | Out-Null
}

function Ensure-PendingBatch([string]$repoRoot, [string]$statePath, [string]$tmpDir, [switch]$generateNext) {
    $promptPath = Get-PendingPromptPathFromState $statePath
    if ($generateNext -or [string]::IsNullOrWhiteSpace($promptPath)) {
        Push-Location $repoRoot
        try {
            $null = & python "scripts/asm_book_loop.py" --state $statePath --tmp-dir $tmpDir next
        }
        finally {
            Pop-Location
        }
        $promptPath = Get-PendingPromptPathFromState $statePath
    }
    if ([string]::IsNullOrWhiteSpace($promptPath)) {
        throw "No pending batch. Run with -GenerateNext or prepare a pending batch first."
    }
    return $promptPath
}

function Get-PendingPrompt([string]$promptPath, [string]$repoRoot) {
    if (-not [System.IO.Path]::IsPathRooted($promptPath)) {
        $promptPath = Join-Path $repoRoot $promptPath
    }
    if (-not (Test-Path $promptPath)) {
        throw "Prompt file not found: $promptPath"
    }
    return @{
        PromptPath = $promptPath
        PromptText = (Get-Content $promptPath -Raw)
    }
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
    $matches = @()
    for ($i = 0; $i -lt $windows.Count; $i++) {
        $window = $windows.Item($i)
        $name = [string]$window.Current.Name
        if ($name -like "*$windowTitle*") {
            $matches += $window
        }
    }
    if ($matches.Count -eq 0) {
        throw "Could not find an open Codex window matching title fragment '$windowTitle'."
    }
    return $matches[0]
}

function Get-BestInputElement($window) {
    $all = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
    for ($i = 0; $i -lt $all.Count; $i++) {
        $node = $all.Item($i)
        if (-not $node.Current.IsEnabled) {
            continue
        }
        if ([string]$node.Current.Name -eq "Terminal input") {
            return $node
        }
    }
    $conditions = New-Object System.Windows.Automation.OrCondition(
        (New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Edit
        )),
        (New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Document
        ))
    )
    $nodes = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, $conditions)
    $best = $null
    for ($i = 0; $i -lt $nodes.Count; $i++) {
        $node = $nodes.Item($i)
        if (-not $node.Current.IsEnabled) {
            continue
        }
        $valuePatternObject = $null
        if ($node.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePatternObject)) {
            $best = $node
        }
    }
    if ($null -ne $best) {
        return $best
    }
    if ($nodes.Count -gt 0) {
        return $nodes.Item($nodes.Count - 1)
    }
    throw "Could not find a writable Codex input control."
}

function Set-ElementText($element, [string]$text) {
    $valuePatternObject = $null
    if ($element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePatternObject)) {
        $valuePattern = [System.Windows.Automation.ValuePattern]$valuePatternObject
        $valuePattern.SetValue($text)
        return $true
    }
    return $false
}

function Invoke-SendButton($window) {
    $buttonCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button
    )
    $buttons = $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition)
    for ($i = 0; $i -lt $buttons.Count; $i++) {
        $button = $buttons.Item($i)
        if (-not $button.Current.IsEnabled) {
            continue
        }
        $name = [string]$button.Current.Name
        if ($name -match "Send|Отправить|Submit") {
            $invoke = $null
            if ($button.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$invoke)) {
                ([System.Windows.Automation.InvokePattern]$invoke).Invoke()
                return $true
            }
        }
    }
    return $false
}

function Send-WithForegroundFallback([string]$windowTitle, [string]$text) {
    Set-Clipboard -Value $text
    $wshell = New-Object -ComObject WScript.Shell
    if (-not $wshell.AppActivate($windowTitle)) {
        throw "Foreground fallback failed: could not activate window '$windowTitle'."
    }
    Start-Sleep -Milliseconds 120
    $wshell.SendKeys("^v")
    Start-Sleep -Milliseconds 60
    $wshell.SendKeys("~")
}

$repoRoot = Resolve-RepoRoot
$statePath = if ([System.IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $repoRoot $State }
$tmpPath = if ([System.IO.Path]::IsPathRooted($TmpDir)) { $TmpDir } else { Join-Path $repoRoot $TmpDir }

$promptPath = Ensure-PendingBatch -repoRoot $repoRoot -statePath $statePath -tmpDir $tmpPath -generateNext:$GenerateNext
$payload = Get-PendingPrompt -promptPath $promptPath -repoRoot $repoRoot
$promptText = [string]$payload.PromptText
$completionMarker = Get-PendingCompletionMarker $statePath
if (-not [string]::IsNullOrWhiteSpace($completionMarker) -and $promptText -notmatch [regex]::Escape($completionMarker)) {
    $promptText = $promptText.TrimEnd() + "`r`n`r`n" + "В самом конце ответа добавь отдельной строкой точный маркер: $completionMarker" + "`r`n"
}

if ($PrintOnly) {
    Write-Output $promptText
    exit 0
}

$window = Find-CodexWindow -windowTitle $WindowTitle
$input = Get-BestInputElement -window $window
$textSet = Set-ElementText -element $input -text $promptText

if ($textSet) {
    Add-Type -AssemblyName System.Windows.Forms
    try {
        $input.SetFocus()
        Start-Sleep -Milliseconds 80
        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        Mark-PendingSentInState $statePath
        Write-Output "Sent pending prompt to Codex via Terminal input focus."
        exit 0
    }
    catch {
    }
    if (Invoke-SendButton -window $window) {
        Mark-PendingSentInState $statePath
        Write-Output "Sent pending prompt to Codex via UI Automation."
        exit 0
    }
}

if ($AllowForegroundFallback) {
    Send-WithForegroundFallback -windowTitle $WindowTitle -text $promptText
    Mark-PendingSentInState $statePath
    Write-Output "Sent pending prompt to Codex via foreground fallback."
    exit 0
}

throw "Background send failed. Input control or send button was not automatable. Re-run with -AllowForegroundFallback if you accept foreground activation."
