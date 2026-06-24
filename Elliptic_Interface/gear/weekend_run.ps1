# ============================================================================
# weekend_run.ps1 -- unattended completion of the 3 CAMWA gap-closing studies,
# then auto git add/commit/push.  Launched by a Windows Scheduled Task so it runs
# independently of the Claude / VS Code windows (survives closing them).
#
# Each study runs ONLY if its JSON output is missing, so anything the live jobs
# already finished is preserved and not recomputed.
# ============================================================================
$ErrorActionPreference = "Continue"
$py   = "C:\Users\NATL\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$git  = "C:\Program Files\Git\cmd\git.exe"
$gear = "C:\Users\NATL\Documents\W-PINN\Elliptic_Interface\Gear_Interface"
$root = "C:\Users\NATL\Documents\W-PINN"
$log  = Join-Path $gear "weekend_run.log"
$env:OMP_NUM_THREADS = "8"; $env:MKL_NUM_THREADS = "8"

function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Out-File -FilePath $log -Append -Encoding utf8 }

Log "=== weekend_run START ==="
Set-Location $gear

$studies = @(
  @{ name = "convergence"; script = "convergence_study.py";    out = "convergence_results.json" },
  @{ name = "baselines";   script = "baselines_gear.py";       out = "baselines_results.json" },
  @{ name = "robustness";  script = "helmholtz_robustness.py"; out = "robustness_results.json" }
)
foreach ($s in $studies) {
  if (Test-Path (Join-Path $gear $s.out)) {
    Log ("SKIP {0} -- {1} already present" -f $s.name, $s.out)
  } else {
    Log ("RUN  {0} : {1}" -f $s.name, $s.script)
    & $py $s.script 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    Log ("DONE {0} (python exit {1})" -f $s.name, $LASTEXITCODE)
  }
}

Log "RUN  make_paper_tables.py"
& $py "make_paper_tables.py" 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
Log ("DONE tables (python exit {0})" -f $LASTEXITCODE)

# ---- auto git add / commit / push ----
Set-Location $root
Log "git add -A"
& $git add -A 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
Log "git commit"
& $git commit -m "Add CAMWA gap-closing studies for gear interface (convergence, baselines, Helmholtz + extreme-contrast robustness)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
Log ("commit exit {0}" -f $LASTEXITCODE)
Log "git push origin main"
& $git push origin main 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
$pushExit = $LASTEXITCODE
Log ("push exit {0}" -f $pushExit)

Log "=== weekend_run END ==="
"weekend_run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  (git push exit=$pushExit -- 0 means pushed OK)" |
  Out-File -FilePath (Join-Path $gear "WEEKEND_DONE.txt") -Encoding utf8

# self-cleanup: remove the scheduled task so it does not fire again
schtasks /delete /tn "WPINN_weekend_run" /f 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
