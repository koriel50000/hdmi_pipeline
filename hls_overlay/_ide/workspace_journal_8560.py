# 2026-09-19T08:08:22.128014600
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

