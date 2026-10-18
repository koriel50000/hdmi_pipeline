# 2026-10-18T17:35:36.895254900
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

vitis.dispose()

