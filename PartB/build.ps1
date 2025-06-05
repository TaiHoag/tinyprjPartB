# PowerShell Build Script for CPU Performance Predictor
# Usage: .\build.ps1 [clean|build|run|help]

param(
    [string]$Action = "build"
)

$ProjectRoot = $PSScriptRoot
$ObjDir = Join-Path $ProjectRoot "obj"
$BinDir = Join-Path $ProjectRoot "bin"
$SrcDir = Join-Path $ProjectRoot "src"
$IncludeDir = Join-Path $ProjectRoot "include"
$Executable = Join-Path $BinDir "cpu_performance_predictor.exe"

# Compiler settings
$CXX = "g++"
$CXXFLAGS = @("-std=c++17", "-Wall", "-Wextra", "-O2")
$IncludeFlag = "-I$IncludeDir"

# Source files
$SourceFiles = @(
    "DataPoint.cpp",
    "Matrix.cpp", 
    "Dataset.cpp",
    "LinearRegression.cpp",
    "Evaluator.cpp"
)

function Show-Help {
    Write-Host "CPU Performance Predictor Build Script" -ForegroundColor Green
    Write-Host "Usage: .\build.ps1 [action]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Cyan
    Write-Host "  build  - Compile the project (default)"
    Write-Host "  clean  - Remove build files"
    Write-Host "  run    - Build and run the program"
    Write-Host "  help   - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Magenta
    Write-Host "  .\build.ps1"
    Write-Host "  .\build.ps1 build"
    Write-Host "  .\build.ps1 clean"
    Write-Host "  .\build.ps1 run"
}

function New-Directories {
    if (-not (Test-Path $ObjDir)) {
        New-Item -ItemType Directory -Path $ObjDir -Force | Out-Null
        Write-Host "Created directory: $ObjDir" -ForegroundColor Green
    }
    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
        Write-Host "Created directory: $BinDir" -ForegroundColor Green
    }
}

function Remove-BuildFiles {
    Write-Host "Cleaning build files..." -ForegroundColor Yellow
    
    if (Test-Path $ObjDir) {
        Remove-Item $ObjDir -Recurse -Force
        Write-Host "Removed: $ObjDir" -ForegroundColor Green
    }
    
    if (Test-Path $BinDir) {
        Remove-Item $BinDir -Recurse -Force
        Write-Host "Removed: $BinDir" -ForegroundColor Green
    }
    
    Write-Host "Clean complete." -ForegroundColor Green
}

function Build-Project {
    Write-Host "Building CPU Performance Predictor..." -ForegroundColor Cyan
    
    # Create directories
    New-Directories
    
    # Compile source files
    $CompileSuccess = $true
    
    foreach ($SourceFile in $SourceFiles) {
        $SourcePath = Join-Path $SrcDir $SourceFile
        $ObjectFile = $SourceFile -replace "\.cpp$", ".o"
        $ObjectPath = Join-Path $ObjDir $ObjectFile
        
        Write-Host "Compiling $SourceFile..." -ForegroundColor Gray
        
        $CompileArgs = $CXXFLAGS + @($IncludeFlag, "-c", $SourcePath, "-o", $ObjectPath)
        $Process = Start-Process -FilePath $CXX -ArgumentList $CompileArgs -Wait -PassThru -NoNewWindow
        
        if ($Process.ExitCode -ne 0) {
            Write-Host "Error compiling $SourceFile" -ForegroundColor Red
            $CompileSuccess = $false
            break
        }
    }
    
    if (-not $CompileSuccess) {
        Write-Host "Build failed!" -ForegroundColor Red
        return $false
    }
    
    # Compile main.cpp
    Write-Host "Compiling main.cpp..." -ForegroundColor Gray
    $MainObjectPath = Join-Path $ObjDir "main.o"
    $MainCompileArgs = $CXXFLAGS + @($IncludeFlag, "-c", "main.cpp", "-o", $MainObjectPath)
    $Process = Start-Process -FilePath $CXX -ArgumentList $MainCompileArgs -Wait -PassThru -NoNewWindow
    
    if ($Process.ExitCode -ne 0) {
        Write-Host "Error compiling main.cpp" -ForegroundColor Red
        return $false
    }
    
    # Link executable
    Write-Host "Linking executable..." -ForegroundColor Gray
    $ObjectFiles = Get-ChildItem $ObjDir -Filter "*.o" | ForEach-Object { $_.FullName }
    $LinkArgs = $CXXFLAGS + $ObjectFiles + @("-o", $Executable)
    $Process = Start-Process -FilePath $CXX -ArgumentList $LinkArgs -Wait -PassThru -NoNewWindow
    
    if ($Process.ExitCode -ne 0) {
        Write-Host "Error linking executable" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "Executable: $Executable" -ForegroundColor Magenta
    return $true
}

function Run-Program {
    if (-not (Test-Path $Executable)) {
        Write-Host "Executable not found. Building first..." -ForegroundColor Yellow
        if (-not (Build-Project)) {
            return
        }
    }
    
    Write-Host "Running CPU Performance Predictor..." -ForegroundColor Cyan
    Write-Host "Executable: $Executable" -ForegroundColor Gray
    Write-Host "Working Directory: $ProjectRoot" -ForegroundColor Gray
    Write-Host ("=" * 60) -ForegroundColor DarkGray
    
    Set-Location $ProjectRoot
    & $Executable
}

# Main script logic
switch ($Action.ToLower()) {
    "clean" {
        Remove-BuildFiles
    }
    "build" {
        Build-Project | Out-Null
    }
    "run" {
        Run-Program
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host "Use '.\build.ps1 help' for usage information." -ForegroundColor Yellow
    }
}
