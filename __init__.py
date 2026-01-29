"""
ComfyUI-UniRig - Automatic rigging and skeleton extraction for ComfyUI.
"""

import os
import traceback

WEB_DIRECTORY = "./web"

# Import nodes
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print(f"[ComfyUI-UniRig] Import failed: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Register web routes
try:
    from server import PromptServer
    from aiohttp import web

    # Static files
    static_path = os.path.join(os.path.dirname(__file__), "web", "static")
    if os.path.exists(static_path):
        PromptServer.instance.app.add_routes([
            web.static('/extensions/ComfyUI-UniRig/static', static_path)
        ])

    # FBX files API
    @PromptServer.instance.routes.get('/unirig/fbx_files')
    async def get_fbx_files(request):
        try:
            from .nodes.nodes_blender.skeleton_io import UniRigLoadRiggedMesh
            source = request.query.get('source_folder', 'output')
            if source == "input":
                files = UniRigLoadRiggedMesh.get_fbx_files_from_input()
            else:
                files = UniRigLoadRiggedMesh.get_fbx_files_from_output()
            return web.json_response(files or [])
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
