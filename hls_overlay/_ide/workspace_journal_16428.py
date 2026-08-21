# 2026-08-21T18:12:12.464863200
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

