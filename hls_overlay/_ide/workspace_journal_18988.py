# 2026-11-03T18:02:29.204952900
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

