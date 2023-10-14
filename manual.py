from ursina import *

class Rubik():
    def __init__(self):

        # plane setup
        Entity(model='quad', scale=60, texture='white_cube', texture_scale=(60, 60),
            rotation_x=90, y=-5, color=color.light_gray)

        # camera setup
        EditorCamera()
        camera.world_position = (0, 0, -15)

        # cubie model and texture
        model, texture = 'models/custom_cube', 'textures/rubik_texture'

        # defining cube sides
        LEFT = {Vec3(-0.5, y, z) for y in [-0.5, 0.5] for z in [-0.5, 0.5]}
        RIGHT = {Vec3(0.5, y, z) for y in [-0.5, 0.5] for z in [-0.5, 0.5]}
        DOWN = {Vec3(x, -0.5, z) for x in [-0.5, 0.5] for z in [-0.5, 0.5]}
        UP = {Vec3(x, 0.5, z) for x in [-0.5, 0.5] for z in [-0.5, 0.5]}
        BACK = {Vec3(x, y, -0.5) for x in [-0.5, 0.5] for y in [-0.5, 0.5]}
        FRONT = {Vec3(x, y, 0.5) for x in [-0.5, 0.5] for y in [-0.5, 0.5]}
        C = LEFT | RIGHT | DOWN | UP | BACK | FRONT 

        # defining transition dictionaries
        self.rotation_axes = {'LEFT': 'x', 'RIGHT': 'x', 'DOWN': 'y', 'UP': 'y', 'BACK': 'z', 'FRONT': 'z'}
        self.cubes_side_positons = {'LEFT': LEFT, 'RIGHT': RIGHT, 'DOWN': DOWN, 'UP': UP, 'BACK': BACK, 'FRONT': FRONT}

        # parameters
        self.animation_time = 0.5
        self.action_trigger = True

        # creating the cubes
        self.PARENT = Entity()
        self.CUBES = [Entity(model=model, texture=texture, position=pos) for pos in C]

    def toggle_trigger(self):
        self.action_trigger = self.action_trigger

    def reparent_to_scene(self):
        for cube in self.CUBES:
            if cube.parent == self.PARENT:
                world_pos, world_rot = round(cube.world_position, 1), cube.world_rotation
                cube.parent = scene
                cube.position, cube.rotation = world_pos, world_rot
        self.PARENT.rotation = 0

    def rotate_side(self, side_name, reverse=False):
        self.toggle_trigger()
        cube_positions = self.cubes_side_positons[side_name]
        rotation_axis = self.rotation_axes[side_name]
        self.reparent_to_scene()
        for cube in self.CUBES:
            if cube.position in cube_positions:
                cube.parent = self.PARENT
                angle = -90 if reverse else 90
                eval(f'self.PARENT.animate_rotation_{rotation_axis}({angle}, duration=self.animation_time)')
        invoke(self.toggle_trigger, delay=self.animation_time+0.1)

    def action(self, key):
        keys = dict(zip('123456', 'LEFT RIGHT DOWN UP BACK FRONT'.split()))
        reverse_keys = dict(zip('qwerty', 'LEFT RIGHT DOWN UP BACK FRONT'.split()))
        if self.action_trigger:
            if key in keys:
                self.rotate_side(keys[key])
            elif key in reverse_keys:
                self.rotate_side(reverse_keys[key], reverse=True)

if __name__ == '__main__':
    app = Ursina(size=(800, 600))
    rubik = Rubik()
    input = lambda key: rubik.action(key)
    app.run()
    