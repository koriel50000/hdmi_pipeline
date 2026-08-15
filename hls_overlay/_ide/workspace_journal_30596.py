# 2026-08-15T13:08:10.530690100
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

comp = client.create_hls_component(name = "hls_component",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

cfg = client.get_config_file(path="E:\workspace\IntelliJ\hdmi_pipeline\hls_overlay\hls_component\hls_config.cfg")

cfg.set_values(key="syn.file", values=["../src/overlay.cpp", "../src/overlay.hpp"])

cfg.set_values(key="tb.file", values=["../src/overlay_test.cpp"])

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

cfg = client.get_config_file(path="/e:/workspace/IntelliJ/hdmi_pipeline/hls_overlay/hls_component/hls_config.cfg")

cfg.set_value(section="hls", key="syn.top", value="pattern_overlay")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="PACKAGE")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="IMPLEMENTATION")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

vitis.dispose()

