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
$form.ClientSize = New-Object System.Drawing.Size(1160, 800)
$form.MinimumSize = New-Object System.Drawing.Size(960, 760)
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

$title = Label 'ATLAS / GreenGPU' 24 20 1100 38
$title.Font = New-Object System.Drawing.Font('Segoe UI', 22, [System.Drawing.FontStyle]::Bold)
$subtitle = Label 'Watch your computer group nearby delivery stops and compare ways to do the job.' 26 65 1100 26
$note = Label 'Experimental project | We measure speed; energy savings have not been proven.' 26 96 1100 24
$note.ForeColor = [System.Drawing.Color]::FromArgb(115, 78, 20)
$buildLabel = Label 'Demo engine' 26 137 110 26
$build = New-Object System.Windows.Forms.ComboBox
$build.Location = New-Object System.Drawing.Point(140, 133)
$build.Size = New-Object System.Drawing.Size(700, 30)
$build.DropDownStyle = 'DropDownList'
$build.Anchor = 'Top, Left, Right'
$form.Controls.Add($build)
$scale = New-Object System.Windows.Forms.ComboBox
$scale.Location = New-Object System.Drawing.Point(855, 133)
$scale.Size = New-Object System.Drawing.Size(279, 30)
$scale.DropDownStyle = 'DropDownList'
$scale.Anchor = 'Top, Right'
$scale.Items.AddRange(@('Small: 2,000 stops', 'Medium: 16,000 stops', 'Large: 64,000 stops'))
$scale.SelectedIndex = 1
$form.Controls.Add($scale)
$quick = Button 'Start live demo' 26 260
$quick.BackColor = [System.Drawing.Color]::FromArgb(20, 112, 83)
$quick.ForeColor = [System.Drawing.Color]::White
$checks = Button 'Check setup' 298 168
$report = Button 'See the result' 478 172
$report.Enabled = $false
$results = Button 'Open results' 662 172
$recorded = Button 'Watch saved demo' 855 279
$recorded.Location = New-Object System.Drawing.Point(855, 231)
$recorded.Anchor = 'Top, Right'
$file = Button 'Use my CSV' 26 140
$file.Top = 231
$stop = Button 'Stop' 174 85
$stop.Top = 231
$stop.Enabled = $false
$monitor = New-Object System.Windows.Forms.CheckBox
$monitor.Text = 'Show machine readings'
$monitor.Checked = $true
$monitor.SetBounds(272, 229, 270, 23)
$form.Controls.Add($monitor)
$description = Label 'Input: generated sample locations' 272 255 560 23
$status = Label 'Ready. Each way of doing the job is tried three times for a fairer comparison.' 26 278 1108 26
$status.Anchor = 'Top, Left, Right'
$status.Font = New-Object System.Drawing.Font('Segoe UI', 10, [System.Drawing.FontStyle]::Bold)
$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(26, 311)
$progress.Size = New-Object System.Drawing.Size(1108, 6)
$progress.Anchor = 'Top, Left, Right'
$form.Controls.Add($progress)
$log = New-Object System.Windows.Forms.TextBox
$log.Location = New-Object System.Drawing.Point(26, 331)
$log.Size = New-Object System.Drawing.Size(1108, 235)
$log.Anchor = 'Top, Bottom, Left, Right'
$log.Multiline = $true
$log.ReadOnly = $true
$log.ScrollBars = 'Both'
$log.WordWrap = $false
$log.Font = New-Object System.Drawing.Font('Consolas', 9)
$log.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($log)
$footer = Label 'Allocation graph (green) shows CPU row share. Performance comparison shows real step time: CPU ONLY (blue), GPU ONLY (orange), ATLAS (green).' 26 756 1108 26
$footer.Anchor = 'Bottom, Left, Right'
$footer.ForeColor = [System.Drawing.Color]::DimGray

$views = New-Object System.Windows.Forms.TabControl
$views.SetBounds(26, 331, 1108, 409)
$views.Anchor = 'Top, Bottom, Left, Right'
$liveTab = New-Object System.Windows.Forms.TabPage('Watch it work')
$machineTab = New-Object System.Windows.Forms.TabPage('Machine readings')
$machineText = New-Object System.Windows.Forms.TextBox
$machineText.Multiline = $true
$machineText.ReadOnly = $true
$machineText.ScrollBars = 'Vertical'
$machineText.Dock = 'Fill'
$machineText.Text = 'Start a task with Show machine readings selected.'
$machineTab.Controls.Add($machineText)
$longRun = New-Object System.Windows.Forms.CheckBox
$longRun.Text = 'Watch longer (512 real steps per try)'
$longRun.Dock = 'Top'
$machineTab.Controls.Add($longRun)
$logTab = New-Object System.Windows.Forms.TabPage('Technical details')
$views.TabPages.AddRange(@($liveTab, $machineTab, $logTab))
$form.Controls.Add($views)
$form.Controls.Remove($log)
$log.Dock = 'Fill'
$logTab.Controls.Add($log)
$liveLayout = New-Object System.Windows.Forms.TableLayoutPanel
$liveLayout.Dock = 'Fill'
$liveLayout.ColumnCount = 1
$liveLayout.RowCount = 2
[void]$liveLayout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 100)))
[void]$liveLayout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, 100)))
[void]$liveLayout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 118)))
$liveTab.Controls.Add($liveLayout)

$chartRow = New-Object System.Windows.Forms.TableLayoutPanel
$chartRow.Dock = 'Fill'
$chartRow.ColumnCount = 3
$chartRow.RowCount = 1
[void]$chartRow.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 33.33)))
[void]$chartRow.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 33.33)))
[void]$chartRow.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 33.34)))
[void]$chartRow.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, 100)))
$liveLayout.Controls.Add($chartRow, 0, 0)

function Live-Chart($title, $column, $axisX, $axisY) {
    $c = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $c.Dock = 'Fill'
    $c.Margin = New-Object System.Windows.Forms.Padding(4)
    $a = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
    $a.AxisX.Title = $axisX
    $a.AxisY.Title = $axisY
    $a.AxisX.MajorGrid.Enabled = $false
    $a.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    [void]$c.ChartAreas.Add($a)
    [void]$c.Titles.Add($title)
    $chartRow.Controls.Add($c, $column, 0)
    return $c
}
$liveMap = Live-Chart 'Locations ready to group' 0 'X coordinate' 'Y coordinate'
$liveTrace = Live-Chart 'CPU workload allocation (%)' 1 'Grouping step' 'CPU row share (%)'
$liveTrace.ChartAreas[0].AxisY.Minimum = 0
$liveTrace.ChartAreas[0].AxisY.Maximum = 100
$liveTrace.ChartAreas[0].AxisX.Minimum = 1
$liveTrace.ChartAreas[0].AxisX.Maximum = 32
$livePerf = Live-Chart 'Performance comparison (ms / iteration)' 2 'Grouping iteration' 'Execution time (ms)'
$livePerf.ChartAreas[0].AxisX.Minimum = 1
$livePerf.ChartAreas[0].AxisX.Maximum = 32
$livePerf.ChartAreas[0].AxisX.IsStartedFromZero = $false
$livePerf.ChartAreas[0].AxisX.MajorGrid.Enabled = $true
$livePerf.ChartAreas[0].AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(240, 240, 240)
$livePerf.ChartAreas[0].AxisY.Minimum = 0
$livePerf.ChartAreas[0].AxisY.Maximum = 1
$livePerf.ChartAreas[0].AxisY.IsStartedFromZero = $true
$livePerf.ChartAreas[0].AxisY.LabelStyle.Format = '0.###'
$perfLegend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend('Modes')
$perfLegend.Docking = 'Top'
$perfLegend.Alignment = 'Center'
$perfLegend.IsDockedInsideChartArea = $false
$perfLegend.Font = New-Object System.Drawing.Font('Segoe UI', 8.5)
[void]$livePerf.Legends.Add($perfLegend)
$liveInfo = New-Object System.Windows.Forms.TextBox
$liveInfo.Dock = 'Fill'
$liveInfo.Margin = New-Object System.Windows.Forms.Padding(4)
$liveInfo.Multiline = $true
$liveInfo.ScrollBars = 'Vertical'
$liveInfo.ReadOnly = $true
$liveInfo.BackColor = [System.Drawing.Color]::White
$liveInfo.Font = New-Object System.Drawing.Font('Consolas', 9)
$liveInfo.Text = "Click Start live demo to try this on your computer now.`r`nWe try CPU ONLY (main processor), GPU ONLY (graphics processor), and ATLAS (CPU + GPU together).`r`nThe groups shown while working are checked before we accept the answer."
$liveLayout.Controls.Add($liveInfo, 0, 1)
$script:liveUpdates = 0

function Reading-Text($value, $unit) {
    if ($null -eq $value) { return 'Unavailable' }
    return ('{0:N2} {1}' -f $value, $unit)
}

function Show-Machine($folder) {
    $path = Join-Path $folder 'machine-live.json'
    if (-not (Test-Path -LiteralPath $path)) { return }
    $d = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $stamp = [DateTimeOffset]::FromUnixTimeMilliseconds([long]([decimal]$d.utc_ns / 1000000))
    $age = ([DateTimeOffset]::UtcNow - $stamp).TotalSeconds
    $label = if ($script:job.HasExited) { 'Last recorded reading' } elseif ($age -gt 3) { 'Waiting for a fresh reading. Last sample' } else { 'Latest reading' }
    $lines = @("$label at $($stamp.ToLocalTime().ToString('HH:mm:ss'))",
        "CPU busy in Linux: $(Reading-Text $d.linux_cpu_percent '%')",
        "ATLAS worker CPU: $(Reading-Text $d.native_cpu_percent_one_core '% of one core')",
        "ATLAS worker memory: $(if ($null -ne $d.native_memory_bytes) { Reading-Text ($d.native_memory_bytes / 1MB) 'MiB' } else { 'Unavailable' })")
    if ($d.trial) { $lines += "Try $($d.trial), Linux process $($d.linux_pid)" }
    else { $lines += 'Sample taken between trials.' }
    if ($d.gpu_note) { $lines += $d.gpu_note }
    foreach ($gpu in $d.gpus) {
        $lines += '', "$($gpu.name) (whole device, including other applications)",
            "Device busy: $(Reading-Text $gpu.'utilization.gpu' '%')",
            "Graphics memory: $(Reading-Text $gpu.'memory.used' 'MiB')",
            "Temperature: $(Reading-Text $gpu.'temperature.gpu' 'C')",
            "Graphics clock: $(Reading-Text $gpu.'clocks.current.graphics' 'MHz')",
            "Reported power: $(Reading-Text $gpu.'power.draw' 'W')"
    }
    $machineText.Text = ($lines -join "`r`n")
}

function Friendly-Mode($name) {
    switch ($name) {
        'CPU only' { return 'CPU ONLY (Main processor)' }
        'GPU only' { return 'GPU ONLY (Graphics processor)' }
        'ATLAS sharing' { return 'ATLAS (CPU + GPU)' }
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
        $liveMap.Series.Clear(); $liveTrace.Series.Clear(); $livePerf.Series.Clear()
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
    $maxRounds = [Math]::Max(2, [int]$d.rounds)
    $liveTrace.Series.Clear()
    $liveTrace.ChartAreas[0].AxisX.Maximum = $maxRounds
    $shares = New-Object System.Windows.Forms.DataVisualization.Charting.Series('CPU share')
    $shares.ChartType = 'StepLine'; $shares.BorderWidth = 3; $shares.Color = [System.Drawing.Color]::SeaGreen
    foreach ($r in $d.trace) { [void]$shares.Points.AddXY([int]$r.round, [double]$r.cpu_percent) }
    [void]$liveTrace.Series.Add($shares)

    $livePerf.Series.Clear()
    $livePerf.ChartAreas[0].AxisX.Maximum = $maxRounds
    $perfSpecs = @(
        @{ Key = 'CPU only'; Label = 'CPU ONLY'; Color = [System.Drawing.Color]::RoyalBlue },
        @{ Key = 'GPU only'; Label = 'GPU ONLY'; Color = [System.Drawing.Color]::DarkOrange },
        @{ Key = 'ATLAS sharing'; Label = 'ATLAS (CPU + GPU)'; Color = [System.Drawing.Color]::SeaGreen }
    )
    $hasPerfPoints = $false
    foreach ($spec in $perfSpecs) {
        $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($spec.Label)
        $series.ChartType = 'Line'
        $series.BorderWidth = 2
        $series.Color = $spec.Color
        $series.Legend = 'Modes'
        $series.IsVisibleInLegend = $true
        $entries = $null
        if ($null -ne $d.performance_traces) {
            $prop = $d.performance_traces.PSObject.Properties[$spec.Key]
            if ($null -ne $prop) { $entries = $prop.Value }
        }
        foreach ($pt in $entries) {
            if ($null -ne $pt.round -and $null -ne $pt.step_ms) {
                [void]$series.Points.AddXY([int]$pt.round, [double]$pt.step_ms)
            }
        }
        if ($series.Points.Count -eq 1) {
            $series.MarkerStyle = 'Circle'; $series.MarkerSize = 6
        } elseif ($series.Points.Count -le 64) {
            $series.MarkerStyle = 'Circle'; $series.MarkerSize = 4
        } else {
            $series.MarkerStyle = 'None'
        }
        if ($series.Points.Count -gt 0) { $hasPerfPoints = $true }
        [void]$livePerf.Series.Add($series)
    }
    $livePerf.ChartAreas[0].AxisY.Maximum = if ($hasPerfPoints) { [Double]::NaN } else { 1 }

    $state = if ($d.phase -in @('validated', 'complete')) { 'Answer checked' } else { 'Working - answer will be checked' }
    $liveInfo.Text = "$state | Try $($d.block) of 3 | Step $($d.round) of $($d.rounds)`r`n"
    if ($d.assignment) {
        $liveInfo.AppendText(("Rows handled: main processor {0:N0}, graphics processor {1:N0}.`r`n" -f $d.assignment.cpu_items, $d.assignment.gpu_items))
    }
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
            if ($monitor.Checked) { $arguments += ' --monitor' }
            if ($longRun.Checked) { $arguments += ' --steps 512' }
            if ($script:inputCsv) { $arguments += ' --windows-input "' + $script:inputCsv + '"' }
        }
        $script:runFolder = $folder
        $script:liveStamp = $null
        $script:liveUpdates = 0
        $script:stopRequested = $false
        $script:fullSuite = $FullSuite
        if (-not $FullSuite) {
            $script:resultFolder = $null
            $views.SelectedTab = $liveTab
            $liveMap.Series.Clear(); $liveTrace.Series.Clear(); $livePerf.Series.Clear()
            $liveInfo.Text = 'Getting your computer ready. The live results will appear here...'
            $machineText.Text = if ($monitor.Checked) { 'Waiting for readings from this run.' } else { 'Machine readings are off for this run.' }
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
        $file.Enabled = $false
        $monitor.Enabled = $false
        $longRun.Enabled = $false
        $stop.Enabled = -not $FullSuite
        $report.Enabled = $false
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
            if ($script:stopRequested -and (Test-Path -LiteralPath $script:runFolder)) {
                [System.IO.File]::WriteAllText((Join-Path $script:runFolder 'stop.request'), '')
            }
            $progressFile = Join-Path $script:runFolder 'progress.json'
            if (Test-Path -LiteralPath $progressFile) {
                $message = (Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json).message
                $status.Text = $message.Replace('CPU only', 'Main processor').Replace('GPU only', 'Graphics processor').Replace('ATLAS sharing', 'Both together').Replace('grouping rounds', 'grouping steps')
            }
            Show-Live $script:runFolder
            if ($monitor.Checked) { Show-Machine $script:runFolder }
            if ($script:stopRequested -and -not $script:job.HasExited) { $status.Text = 'Stopping the task...' }
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
                if ($SmokeTest) { Open-RecordedDemo $script:runFolder }
                $script:resultFolder = $script:runFolder
                $report.Enabled = $true
                $status.Text = 'All answers matched. Select See the result to inspect any step or row.'
            }
        } else {
            $liveInfo.Text = if ($script:stopRequested) { 'Run stopped. Partial records are in Open results.' } else { 'The task could not finish. See Technical details for the reason.' }
            $status.Text = $liveInfo.Text
            $views.SelectedTab = $logTab
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
        $scale.Enabled = -not $script:inputCsv
        $recorded.Enabled = $true
        $file.Enabled = $true
        $monitor.Enabled = $true
        $longRun.Enabled = $true
        $stop.Enabled = $false
        $report.Enabled = [bool]$script:resultFolder
    }
})
$quick.Add_Click({ Start-Demo $false })
$checks.Add_Click({ Start-Demo $true })
$stop.Add_Click({ $script:stopRequested = $true; $stop.Enabled = $false })
$file.Add_Click({
    if ($script:inputCsv) {
        $script:inputCsv = $null
        $description.Text = 'Input: generated sample locations'
        $file.Text = 'Use my CSV'
        $scale.Enabled = $true
        return
    }
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $dialog.Filter = 'CSV files (*.csv)|*.csv'
        if ($dialog.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) {
            $script:inputCsv = $dialog.FileName
            $description.Text = 'Input: ' + $dialog.SafeFileName
            $file.Text = 'Use sample data'
            $scale.Enabled = $false
        }
    } finally { $dialog.Dispose() }
})
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
        Open-RecordedDemo $script:resultFolder
    } catch { $log.Text = $_.Exception.Message; $views.SelectedTab = $logTab }
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
        if (-not $script:fullSuite) { $script:stopRequested = $true; $stop.Enabled = $false }
        $status.Text = 'Waiting for the current task to finish or stop.'
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
