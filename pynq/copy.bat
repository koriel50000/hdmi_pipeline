@echo off

set "BAT_DIR=%~dp0"
cd /d "%BAT_DIR%"

copy /y "..\hdmi_pipeline.gen\sources_1\bd\design_1\hw_handoff\design_1.hwh" .
copy /y "..\hdmi_pipeline.runs\impl_1\design_1_wrapper.bit" design_1.bit
