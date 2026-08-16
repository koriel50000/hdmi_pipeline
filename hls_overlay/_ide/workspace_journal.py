# 2026-08-16T11:58:42.436714300
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

cfg = client.get_config_file(path="E:\workspace\IntelliJ\hdmi_pipeline\hls_overlay\hls_component\hls_config.cfg")

cfg.set_values(key="syn.file", values=["../src/overlay.cpp", "../src/overlay.hpp", "../src/image.hpp"])

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

