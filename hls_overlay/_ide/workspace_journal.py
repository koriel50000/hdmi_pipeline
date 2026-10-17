# 2026-10-17T09:03:21.145813400
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

