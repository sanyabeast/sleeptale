@echo off
setlocal enabledelayedexpansion

:: SleepTale Interactive Menu
:: This batch file provides an interactive menu to control all processes
:: required for sleep story video generation

:menu
cls
echo.
echo ===================================================
echo           SLEEPTALE INTERACTIVE MENU
echo ===================================================
echo.
echo  1. Generate Complete Videos (Full Pipeline)
echo  2. Generate Stories Only
echo  3. Generate Voice Lines Only
echo  4. Generate Videos from Existing Voice Lines
echo  5. View Configuration
echo  6. Exit
echo.
echo ===================================================
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto full_pipeline
if "%choice%"=="2" goto generate_stories
if "%choice%"=="3" goto generate_voice_lines
if "%choice%"=="4" goto generate_videos
if "%choice%"=="5" goto view_config
if "%choice%"=="6" goto end

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:full_pipeline
cls
echo.
echo ===================================================
echo           GENERATE COMPLETE VIDEOS
echo ===================================================
echo.
echo This will run the complete pipeline:
echo 1. Generate stories
echo 2. Create voice lines
echo 3. Produce final videos
echo.

set story_count=1
set /p story_count="Number of stories to generate [1] (or 'b' to go back): "
if /i "!story_count!"=="b" goto menu
if "!story_count!"=="" set story_count=1

set quality=720
set /p quality="Video quality (360, 480, 720, 1080) [720] (or 'b' to go back): "
if /i "!quality!"=="b" goto menu
if "!quality!"=="" set quality=720

set duration=10
set /p duration="Story duration in minutes [10] (or 'b' to go back): "
if /i "!duration!"=="b" goto menu
if "!duration!"=="" set duration=10

echo.
echo Running full pipeline with:
echo - !story_count! stories
echo - !quality!p video quality
echo - !duration! minutes duration
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

python make.py -c !story_count! -q !quality! -d !duration!

echo.
echo Process completed. Press any key to return to menu...
pause >nul
goto menu

:generate_stories
cls
echo.
echo ===================================================
echo           GENERATE STORIES ONLY
echo ===================================================
echo.

set story_count=1
set /p story_count="Number of stories to generate [1] (or 'b' to go back): "
if /i "!story_count!"=="b" goto menu
if "!story_count!"=="" set story_count=1

set duration=10
set /p duration="Story duration in minutes [10] (or 'b' to go back): "
if /i "!duration!"=="b" goto menu
if "!duration!"=="" set duration=10

set model=""
set /p model="Model to use (leave empty for default from config, 'b' to go back): "
if /i "!model!"=="b" goto menu

echo.
echo Generating !story_count! stories with !duration! minutes duration...
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

if "!model!"=="" (
    python make_story.py -c !story_count! -d !duration!
) else (
    python make_story.py -c !story_count! -d !duration! -m !model!
)

echo.
echo Process completed. Press any key to return to menu...
pause >nul
goto menu

:generate_voice_lines
cls
echo.
echo ===================================================
echo           GENERATE VOICE LINES ONLY
echo ===================================================
echo.
echo This will create voice lines for existing stories.
echo.

set story_name=""
set /p story_name="Story name (leave empty for all stories, 'b' to go back): "
if /i "!story_name!"=="b" goto menu

echo.
if "!story_name!"=="" (
    echo Generating voice lines for all stories...
    python make_voice_lines.py
) else (
    echo Generating voice lines for story: !story_name!
    python make_voice_lines.py -s !story_name!
)

echo.
echo Process completed. Press any key to return to menu...
pause >nul
goto menu

:generate_videos
cls
echo.
echo ===================================================
echo           GENERATE VIDEOS ONLY
echo ===================================================
echo.
echo This will create videos from existing voice lines.
echo.

set story_name=""
set /p story_name="Story name (leave empty for all stories, 'b' to go back): "
if /i "!story_name!"=="b" goto menu

set quality=720
set /p quality="Video quality (360, 480, 720, 1080) [720] (or 'b' to go back): "
if /i "!quality!"=="b" goto menu
if "!quality!"=="" set quality=720

set force=""
set /p force="Force regeneration of existing videos? (y/n) [n] (or 'b' to go back): "
if /i "!force!"=="b" goto menu
if /i "!force!"=="y" (
    set force=-f
) else (
    set force=
)

echo.
if "!story_name!"=="" (
    echo Generating videos for all stories with quality: !quality!p
    python make_video.py -q !quality! !force!
) else (
    echo Generating video for story: !story_name! with quality: !quality!p
    python make_video.py -s !story_name! -q !quality! !force!
)

echo.
echo Process completed. Press any key to return to menu...
pause >nul
goto menu

:view_config
cls
echo.
echo ===================================================
echo           CURRENT CONFIGURATION
echo ===================================================
echo.
echo Displaying current configuration from config.yaml:
echo.

type config.yaml

echo.
echo Press any key to return to menu...
pause >nul
goto menu

:end
cls
echo.
echo Thank you for using SleepTale Interactive Menu.
echo Goodbye!
echo.
exit /b 0
