# 2026-09-23T08:42:14.194458200
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

vitis.dispose()

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="PACKAGE")

