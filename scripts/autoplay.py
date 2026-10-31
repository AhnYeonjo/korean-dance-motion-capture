import bpy
def play():
    w = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=w, screen=w.screen):
        bpy.ops.screen.animation_play()
bpy.app.timers.register(play, first_interval=2)
