# 2026-08-15T18:46:38.876453900
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

cfg = client.get_config_file(path="/e:/workspace/IntelliJ/hdmi_pipeline/hls_overlay/hls_component/hls_config.cfg")

cfg.set_value(key="part", value="xc7z020clg400-1")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

vitis.dispose()

