# 2026-08-19T22:07:54.675498300
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_overlay")

vitis.dispose()

