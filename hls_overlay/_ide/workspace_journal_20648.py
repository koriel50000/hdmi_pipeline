# 2026-09-16T22:36:53.233490700
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

