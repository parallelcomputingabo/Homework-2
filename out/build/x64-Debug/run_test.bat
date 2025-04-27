@echo off
setlocal enabledelayedexpansion

:: Output file
set OUTPUT_FILE=results.txt

:: Clear previous results
echo Test Results > %OUTPUT_FILE%
echo ========================= >> %OUTPUT_FILE%

:: Loop through modes 0 to 9
for /L %%M in (0,1,9) do (
    echo Running mode %%M >> %OUTPUT_FILE%
    echo ------------------------- >> %OUTPUT_FILE%
    
    :: Run 3 times for each mode
    for /L %%R in (1,1,5) do (
        echo Run %%R >> %OUTPUT_FILE%
        matmul.exe %%M >> %OUTPUT_FILE%
        echo. >> %OUTPUT_FILE%
    )
    
    echo. >> %OUTPUT_FILE%
)

echo All runs complete.
pause
