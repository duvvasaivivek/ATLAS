param([switch]$SmokeTest)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
[System.Windows.Forms.Application]::EnableVisualStyles()
$root = Split-Path -Parent $PSScriptRoot
$script:job = $null
$script:lastExit = $null
$script:lastLog = $null

$form = New-Object System.Windows.Forms.Form
$form.Text = 'ATLAS | GreenGPU Demo'
$form.ClientSize = New-Object System.Drawing.Size(860, 800)
$form.MinimumSize = New-Object System.Drawing.Size(876, 839)
$form.StartPosition = 'CenterScreen'
$form.BackColor = [System.Drawing.Color]::FromArgb(245, 247, 250)
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

function Label($text, $x, $y, $width, $height) {
    $control = New-Object System.Windows.Forms.Label
    $control.Text = $text
    $control.Location = New-Object System.Drawing.Point($x, $y)
    $control.Size = New-Object System.Drawing.Size($width, $height)
    $form.Controls.Add($control)
    return $control
}
function Button($text, $x, $width) {
    $control = New-Object System.Windows.Forms.Button
    $control.Text = $text
    $control.Location = New-Object System.Drawing.Point($x, 175)
    $control.Size = New-Object System.Drawing.Size($width, 42)
    $control.FlatStyle = 'Flat'
    $control.BackColor = [System.Drawing.Color]::White
    $form.Controls.Add($control)
    return $control
}

$title = Label 'ATLAS / GreenGPU' 24 20 800 38
$title.Font = New-Object System.Drawing.Font('Segoe UI', 22, [System.Drawing.FontStyle]::Bold)
$subtitle = Label 'Watch your computer group nearby delivery stops and compare ways to do the job.' 26 65 800 26
$note = Label 'Experimental project | We measure speed; energy savings have not been proven.' 26 96 800 24
$note.ForeColor = [System.Drawing.Color]::FromArgb(115, 78, 20)
$buildLabel = Label 'Demo engine' 26 137 110 26
$build = New-Object System.Windows.Forms.ComboBox
$build.Location = New-Object System.Drawing.Point(140, 133)
$build.Size = New-Object System.Drawing.Size(400, 30)
$build.DropDownStyle = 'DropDownList'
$build.Anchor = 'Top, Left, Right'
$form.Controls.Add($build)
$scale = New-Object System.Windows.Forms.ComboBox
$scale.Location = New-Object System.Drawing.Point(555, 133)
$scale.Size = New-Object System.Drawing.Size(279, 30)
$scale.DropDownStyle = 'DropDownList'
$scale.Items.AddRange(@('Small: 2,000 stops', 'Medium: 16,000 stops', 'Large: 64,000 stops'))
$scale.SelectedIndex = 1
$form.Controls.Add($scale)
$quick = Button 'Start live demo' 26 260
$quick.BackColor = [System.Drawing.Color]::FromArgb(20, 112, 83)
$quick.ForeColor = [System.Drawing.Color]::White
$checks = Button 'Check setup' 298 168
$report = Button 'Full report' 478 172
$results = Button 'Open results' 662 172
$recorded = Button 'Watch saved demo' 555 279
$recorded.Location = New-Object System.Drawing.Point(555, 231)
$description = Label 'Choose how many stops to group, then click Start live demo. Watch the work happen below.' 26 231 510 42
$description.Anchor = 'Top, Left, Right'
$status = Label 'Ready. Each way of doing the job is tried three times for a fairer comparison.' 26 278 808 26
$status.Anchor = 'Top, Left, Right'
$status.Font = New-Object System.Drawing.Font('Segoe UI', 10, [System.Drawing.FontStyle]::Bold)
$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(26, 311)
$progress.Size = New-Object System.Drawing.Size(808, 6)
$progress.Anchor = 'Top, Left, Right'
$form.Controls.Add($progress)
$log = New-Object System.Windows.Forms.TextBox
$log.Location = New-Object System.Drawing.Point(26, 331)
$log.Size = New-Object System.Drawing.Size(808, 235)
$log.Anchor = 'Top, Bottom, Left, Right'
$log.Multiline = $true
$log.ReadOnly = $true
$log.ScrollBars = 'Both'
$log.WordWrap = $false
$log.Font = New-Object System.Drawing.Font('Consolas', 9)
$log.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($log)
$footer = Label 'The green line shows the main processor''s share; graphics handles the rest. Display updates also take time.' 26 756 808 26
$footer.Anchor = 'Bottom, Left, Right'
$footer.ForeColor = [System.Drawing.Color]::DimGray

$views = New-Object System.Windows.Forms.TabControl
$views.SetBounds(26, 331, 808, 409)
$views.Anchor = 'Top, Bottom, Left, Right'
$liveTab = New-Object System.Windows.Forms.TabPage('Watch it work')
$logTab = New-Object System.Windows.Forms.TabPage('Details')
$views.TabPages.AddRange(@($liveTab, $logTab))
$form.Controls.Add($views)
$form.Controls.Remove($log)
$log.Dock = 'Fill'
$logTab.Controls.Add($log)
function Live-Chart($title, $x, $axisX, $axisY) {
    $c = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $c.SetBounds($x, 0, 395, 250)
    $a = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
    $a.AxisX.Title = $axisX
    $a.AxisY.Title = $axisY
    $a.AxisX.MajorGrid.Enabled = $false
    $a.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    [void]$c.ChartAreas.Add($a)
    [void]$c.Titles.Add($title)
    $liveTab.Controls.Add($c)
    return $c
}
$liveMap = Live-Chart 'Delivery stops: ready to start' 0 'Across the example area' 'Up the example area'
$liveTrace = Live-Chart 'How the work is shared' 400 'Grouping step' 'Main processor''s share (%)'
$liveTrace.ChartAreas[0].AxisY.Minimum = 0
$liveTrace.ChartAreas[0].AxisY.Maximum = 100
$liveTrace.ChartAreas[0].AxisX.Minimum = 1
$liveTrace.ChartAreas[0].AxisX.Maximum = 32
$liveInfo = New-Object System.Windows.Forms.TextBox
$liveInfo.SetBounds(10, 255, 775, 113)
$liveInfo.Multiline = $true
$liveInfo.ReadOnly = $true
$liveInfo.BackColor = [System.Drawing.Color]::White
$liveInfo.Font = New-Object System.Drawing.Font('Consolas', 9)
$liveInfo.Text = "Click Start live demo to try this on your computer now.`r`nWe try the main processor, the graphics processor, then both together.`r`nThe groups shown while working are checked before we accept the answer."
$liveTab.Controls.Add($liveInfo)
$script:liveUpdates = 0

function Friendly-Mode($name) {
    switch ($name) {
        'CPU only' { return 'Main processor alone' }
        'GPU only' { return 'Graphics processor alone' }
        'ATLAS sharing' { return 'Both working together' }
        default { return $name }
    }
}

function Show-Live($folder) {
    $path = Join-Path $folder 'live.json'
    if (-not (Test-Path -LiteralPath $path)) { return }
    $stamp = (Get-Item -LiteralPath $path).LastWriteTimeUtc.Ticks
    if ($script:liveStamp -eq $stamp) { return }
    $d = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $script:liveStamp = $stamp
    $script:liveUpdates++
    if ($d.phase -eq 'failed') {
        $liveMap.Series.Clear(); $liveTrace.Series.Clear()
        $liveInfo.Text = 'RUN FAILED. No successful comparison published.' + "`r`n" + $d.error
        return
    }
    $liveMap.Series.Clear()
    $colors = @('SteelBlue', 'DarkOrange', 'SeaGreen', 'MediumPurple')
    for ($group = 0; $group -lt 4; $group++) {
        $s = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Area ' + ($group + 1))
        $s.ChartType = 'Point'; $s.MarkerSize = 4
        $s.Color = if ($null -eq $d.labels) { [System.Drawing.Color]::Gray } else { [System.Drawing.Color]::FromName($colors[$group]) }
        for ($i = 0; $i -lt $d.points.Count; $i++) {
            if (($null -eq $d.labels -and $group -eq 0) -or ($null -ne $d.labels -and $d.labels[$i] -eq $group)) {
                [void]$s.Points.AddXY([double]$d.points[$i][0], [double]$d.points[$i][1])
            }
        }
        [void]$liveMap.Series.Add($s)
    }
    $centers = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Current group centers')
    $centers.ChartType = 'Point'; $centers.MarkerStyle = 'Cross'; $centers.MarkerSize = 12; $centers.Color = [System.Drawing.Color]::Black
    foreach ($p in $d.centroids) { [void]$centers.Points.AddXY([double]$p[0], [double]$p[1]) }
    [void]$liveMap.Series.Add($centers)
    $phaseText = switch ($d.phase) {
        'starting' { 'getting ready' }
        'computing' { 'working' }
        'validated' { 'answer checked' }
        'complete' { 'finished' }
        default { $d.phase }
    }
    $liveMap.Titles[0].Text = "$(Friendly-Mode $d.mode): $phaseText"
    $liveTrace.Series.Clear()
    $shares = New-Object System.Windows.Forms.DataVisualization.Charting.Series('CPU share')
    $shares.ChartType = 'StepLine'; $shares.BorderWidth = 3; $shares.Color = [System.Drawing.Color]::SeaGreen
    foreach ($r in $d.trace) { [void]$shares.Points.AddXY([int]$r.round, [double]$r.cpu_percent) }
    [void]$liveTrace.Series.Add($shares)
    $state = if ($d.phase -in @('validated', 'complete')) { 'Answer checked' } else { 'Working - answer will be checked' }
    $liveInfo.Text = "$state | Try $($d.block) of 3 | Step $($d.round) of 32`r`n"
    foreach ($name in @('CPU only', 'GPU only', 'ATLAS sharing')) {
        $rows = @($d.completed | Where-Object mode -eq $name)
        if ($rows.Count) {
            $latest = $rows[-1]
            $liveInfo.AppendText(('{0}: {1}/3 checked; grouping {2:N2} ms, whole task {3:N2} ms' -f (Friendly-Mode $name), $rows.Count, $latest.workload_ms, $latest.command_ms) + "`r`n")
        }
    }
}

$builds = @(Get-ChildItem -LiteralPath (Join-Path $root 'build') -Directory -ErrorAction SilentlyContinue |
    Where-Object { (Test-Path -LiteralPath (Join-Path $_.FullName 'atlas_kmeans')) -and
                   (Test-Path -LiteralPath (Join-Path $_.FullName 'CTestTestfile.cmake')) } |
    Sort-Object LastWriteTime -Descending)
foreach ($item in $builds) { [void]$build.Items.Add('build/' + $item.Name) }
if ($build.Items.Count) {
    $build.SelectedIndex = 0
} else {
    $quick.Enabled = $false
    $checks.Enabled = $false
    $status.Text = 'No native build found. Follow the WSL setup in README.md first.'
}
$log.Text = "WHAT ATLAS SOLVES`r`nA GPU-only job can leave CPU capacity unused. Sharing work may help, but moving data and coordinating processors costs time.`r`n`r`nWHAT TO SHOW`r`n1. Thousands of delivery stops become four nearby groups.`r`n2. Compare CPU-only, GPU-only and ATLAS on identical input.`r`n3. Show how ATLAS changes the split; explain which approach won.`r`n`r`nWHAT THE AUDIENCE CAN CONCLUDE`r`nThe answers match. The times are measured. Sharing is not always faster.`r`nThis demonstrates grouping, not route optimization or energy savings.`r`n`r`nRun a fresh comparison with WSL/CUDA, or open a recorded demo offline."

function Open-RecordedDemo($folder) {
    $inventory = Get-Content -LiteralPath (Join-Path $folder 'checksums.json') -Raw | ConvertFrom-Json
    if (-not $inventory.PSObject.Properties['demo.html']) { throw 'This record has no verified presentation.' }
    foreach ($entry in $inventory.PSObject.Properties) {
        if ([System.IO.Path]::GetFileName($entry.Name) -ne $entry.Name) { throw 'Invalid evidence path.' }
        if ((Get-FileHash -LiteralPath (Join-Path $folder $entry.Name) -Algorithm SHA256).Hash -ne $entry.Value) { throw 'Evidence checksum mismatch. Record was not opened.' }
    }
    if (-not $SmokeTest) { Start-Process -FilePath (Join-Path $folder 'demo.html') }
}

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 250

function Start-Demo([bool]$FullSuite) {
    if ($script:job -or $build.SelectedIndex -lt 0) { return }
    try {
        $wsl = (Get-Command wsl.exe -ErrorAction Stop).Source
        $chosen = [string]$build.SelectedItem
        $folder = Join-Path $root ('results/demo/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
        $prefix = '--cd "' + $root + '" --exec env OMP_NUM_THREADS=4 '
        if ($FullSuite) {
            $arguments = $prefix + 'ctest --test-dir "' + $chosen + '" --output-on-failure --no-tests=error'
            [void](New-Item -ItemType Directory -Path $folder)
        } else {
            $n = @(2000, 16000, 64000)[$scale.SelectedIndex]
            $relative = 'results/demo/' + (Split-Path -Leaf $folder)
            $arguments = $prefix + 'python3 scripts/demo_workload.py --build "' + $chosen + '" --size ' + $n + ' --output "' + $relative + '"'
        }
        $script:runFolder = $folder
        $script:liveStamp = $null
        $script:liveUpdates = 0
        $script:fullSuite = $FullSuite
        if (-not $FullSuite) {
            $views.SelectedTab = $liveTab
            $liveMap.Series.Clear(); $liveTrace.Series.Clear()
            $liveInfo.Text = 'Getting your computer ready. The live results will appear here...'
        } else { $views.SelectedTab = $logTab }
        $script:lastLog = Join-Path $folder 'native-checks.txt'
        $command = 'wsl.exe ' + $arguments
        $script:logHeader = "ATLAS demo | $(Get-Date -Format o)`r`n$command`r`n`r`n"
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $wsl
        $info.Arguments = $arguments
        $info.WorkingDirectory = $root
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        [void]$process.Start()
        $script:stdout = $process.StandardOutput.ReadToEndAsync()
        $script:stderr = $process.StandardError.ReadToEndAsync()
        $script:job = $process
        $script:lastExit = $null
        $quick.Enabled = $false
        $checks.Enabled = $false
        $build.Enabled = $false
        $scale.Enabled = $false
        $recorded.Enabled = $false
        $progress.Style = 'Marquee'
        $status.Text = 'Your computer is working on the example now...'
        $status.ForeColor = [System.Drawing.Color]::FromArgb(20, 112, 83)
        $log.Text = $script:logHeader + "Live computation is shown in the first tab. Final presentation opens after correctness checks pass."
        $timer.Start()
    } catch {
        $status.Text = 'Could not start the demo.'
        $log.Text = $_.Exception.Message + "`r`nCheck that WSL is installed and starts normally."
        $script:lastExit = 1
    }
}

$timer.Add_Tick({
    if ($script:job -and -not $script:fullSuite) {
        try {
            $progressFile = Join-Path $script:runFolder 'progress.json'
            if (Test-Path -LiteralPath $progressFile) {
                $message = (Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json).message
                $status.Text = $message.Replace('CPU only', 'Main processor').Replace('GPU only', 'Graphics processor').Replace('ATLAS sharing', 'Both together').Replace('grouping rounds', 'grouping steps')
            }
            Show-Live $script:runFolder
        } catch { } # Progress may be replaced atomically between polling and reading.
    }
    if (-not $script:job -or -not $script:job.HasExited) { return }
    if (-not $script:stdout.IsCompleted -or -not $script:stderr.IsCompleted) { return }
    $timer.Stop()
    try {
        $script:lastExit = $script:job.ExitCode
        # WSL startup diagnostics can be UTF-16 even though child output is UTF-8.
        $output = ($script:stdout.Result + "`r`n" + $script:stderr.Result).Replace([string][char]0, '')
        $log.Text = $script:logHeader + $output.Replace("`r`n", "`n").Replace("`n", "`r`n")
        $log.Text += "`r`nExit code: $script:lastExit`r`nSaved: $script:lastLog"
        if (-not (Test-Path -LiteralPath $script:runFolder)) { [void](New-Item -ItemType Directory -Path $script:runFolder -Force) }
        $log.Text | Set-Content -LiteralPath $script:lastLog -Encoding UTF8
        if ($script:lastExit -eq 0) {
            if ($script:fullSuite) { $status.Text = 'Setup checks passed. The computer completed the test jobs correctly.' }
            else {
                Show-Live $script:runFolder
                Open-RecordedDemo $script:runFolder
                $status.Text = 'Finished! All three ways gave the same answer. See which was fastest.'
            }
        } else {
            $liveInfo.Text = 'RUN FAILED. See Logs and explanation; no successful comparison published.'
            $status.Text = 'FAILED | See the output below and the saved log.'
            $status.ForeColor = [System.Drawing.Color]::Firebrick
        }
    } catch {
        $script:lastExit = 1
        $status.Text = 'Could not collect the results.'
        $log.AppendText("`r`n" + $_.Exception.Message)
    } finally {
        $script:job.Dispose()
        $script:job = $null
        $progress.Style = 'Blocks'
        $quick.Enabled = $true
        $checks.Enabled = $true
        $build.Enabled = $true
        $scale.Enabled = $true
        $recorded.Enabled = $true
    }
})
$quick.Add_Click({ Start-Demo $false })
$checks.Add_Click({ Start-Demo $true })
$recorded.Add_Click({
    try {
        $latest = Get-ChildItem -LiteralPath (Join-Path $root 'results/demo') -Filter demo.html -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latest) { throw 'No recorded presentation yet. Run a comparison once with WSL/GPU available.' }
        Open-RecordedDemo $latest.DirectoryName
        $script:lastLog = Join-Path $latest.DirectoryName 'native-checks.txt'
        $status.Text = 'Recorded demonstration opened. Its measurement date is shown on the page.'
    } catch { $log.Text = $_.Exception.Message }
})
$report.Add_Click({
    try {
        $latest = Get-ChildItem -LiteralPath (Join-Path $root 'results') -Filter report.html -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latest) { throw 'No report found. Generate one with scripts/validate_release.py in WSL; see README.md.' }
        Start-Process -FilePath $latest.FullName
    } catch { $log.Text = $_.Exception.Message }
})
$results.Add_Click({
    try {
        $destination = Join-Path $root 'results'
        if ($script:lastLog) { $destination = Split-Path -Parent $script:lastLog }
        Start-Process explorer.exe -ArgumentList ('"' + $destination + '"')
    } catch { $log.Text = $_.Exception.Message }
})
$form.Add_FormClosing({
    param($sender, $eventArgs)
    if ($script:job) {
        $eventArgs.Cancel = $true
        $status.Text = 'Checks are still running. Close the window after they finish.'
    }
})

if ($SmokeTest) {
    # Exercise the button handler and asynchronous completion without showing a window.
    if (-not $quick.Enabled) { throw 'Smoke test requires a configured native build.' }
    $scale.SelectedIndex = 0
    Start-Demo $false
    while ($script:job) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 50
    }
    Write-Output $log.Text
    Write-Output "Live dashboard updates: $script:liveUpdates"
    if ($script:lastExit -eq 0 -and $script:liveUpdates -lt 1) { throw 'No live computation snapshots reached the dashboard.' }
    $timer.Dispose()
    $form.Dispose()
    exit $script:lastExit
}
try { [void]$form.ShowDialog() }
finally { $timer.Dispose(); $form.Dispose() }
