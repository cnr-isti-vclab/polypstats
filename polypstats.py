
import pygame

import pymeshlab
from pygame.locals import *

from OpenGL.GL import glDrawElements
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image
import ctypes

import glm

import imgui
from imgui.integrations.pygame import PygameRenderer

import os
import ctypes
import numpy as np
import trackball 
import texture
import metashape_loader
import fbo
from  shaders import vertex_shader, fragment_shader, vertex_shader_fsq, fragment_shader_fsq,bbox_shader_str
import maskout
import time
import metrics
from plane import fit_plane, project_point_on_plane
from ctypes import c_uint32, cast, POINTER


import sys 

from  detector import apply_yolo
import numpy as np
import pandas as pd
import os
from collections import Counter



def check_gl_errors():
    """Check for OpenGL errors and print them if any exist."""
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {gluErrorString(error).decode('utf-8')}")


class renderable:
     def __init__(self, vao, n_verts, n_faces,texture_id,mask_id):
         self.vao = vao
         self.n_verts = n_verts
         self.n_faces = n_faces
         self.texture_id = texture_id
         self.mask_id = mask_id
         
     vao = None #vertex array object
     n_verts = None
     n_faces = None

class shader:
    def __init__(self, vertex_shader_str , fragment_shader_str):
        self.uniforms = {}
        self.program =compileProgram(
        compileShader(vertex_shader_str, GL_VERTEX_SHADER),
        compileShader(fragment_shader_str, GL_FRAGMENT_SHADER)
        )
    def uni(self, name):
            if name not in self.uniforms:
                self.uniforms[name] = glGetUniformLocation(self.program, name)
            return self.uniforms[name]
    
def create_buffers_frame():
    
    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    verts = [0,0,0, 1,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,1]
    verts = np.array(verts, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,verts.nbytes, verts, GL_STATIC_DRAW)
    
    # Generate buffers to hold our vertices
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aColor')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    col = [1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1]
    col = np.array(col, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,col.nbytes, col, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object
     
def create_buffers_fsq():
        
    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader_fsq.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    verts = [-1,-1,0, 1,-1,0, 1,1,0, -1,-1,0, 1,1,0, -1,1,0] 
    verts = np.array(verts, dtype=np.float32)
 
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,verts.nbytes, verts, GL_STATIC_DRAW)
    
    # Unbind other stuff
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object

def create_buffers(verts,wed_tcoord,inds,shader0):
    global color_buffer
    vert_pos            = np.zeros((len(inds) * 3,  3), dtype=np.float32)
    tcoords             = np.zeros((len(inds) * 3,  2), dtype=np.float32)
    maskout.tri_color   = np.zeros((len(vert_pos)  ,3), dtype=np.float32)
    for i in range(len(inds)):
        vert_pos[i*3] = verts[inds[i,0]]
        vert_pos[i*3+1] = verts[inds[i,1]]
        vert_pos[i*3+2] = verts[inds[i,2]]

        tcoords [i * 3  ] = wed_tcoord[i*3   ]
        tcoords [i * 3+1] = wed_tcoord[i*3+1 ]
        tcoords [i * 3+2] = wed_tcoord[i*3+2 ]

    vert_pos = vert_pos.flatten()
    tcoords = tcoords.flatten()
    maskout.tri_color  = maskout.tri_color.flatten()   

    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,vert_pos.nbytes, vert_pos, GL_STATIC_DRAW)
    

    # color
    
   
    color_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aColor')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,maskout.tri_color.nbytes, maskout.tri_color, GL_STATIC_DRAW)



    # Generate buffers to hold our texcoord
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'texcoord' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aTexCoord')
    glEnableVertexAttribArray(position)
    
    # Describe the texcoord data layout in the buffer
    glVertexAttribPointer(position, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,tcoords.nbytes, tcoords, GL_STATIC_DRAW)

    # Create an array of n*3 elements as described
    n = len(vert_pos)
    triangle_ids = np.repeat(np.arange(n), 3).astype(np.float32).reshape(-1, 3).flatten()

    # Generate buffers to hold our triangle ids
    triangle_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer)

    # Get the position of the 'aIdTriangle' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aIdTriangle')
    glEnableVertexAttribArray(position)

    # Describe the triangle id data layout in the buffer
    glVertexAttribPointer(position, 1, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER, triangle_ids.nbytes, triangle_ids, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    return vertex_array_object

def display_image():
        global id_node
        global mask_zoom
        global mask_xpos
        global mask_ypos
        global W
        global H
        global curr_zoom
        global curr_center
        global curr_tra
        global tra_xstart
        global tra_ystart
        global show_mask

        current_unit = glGetIntegerv(GL_ACTIVE_TEXTURE)
        current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glClearColor (1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

       
        mask = maskout.all_masks.nodes[id_node].mask

        glUseProgram(shader_fsq.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniform1i(shader_fsq.uni("uColorTex"),0)

        # Get the currently bound texture on GL_TEXTURE_2D

        glActiveTexture(GL_TEXTURE8)
        current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
        glBindTexture(GL_TEXTURE_2D, mask.id_texture)
        glUniform1i(shader_fsq.uni("uMask"),8)
        glUniform2i(shader_fsq.uni("uOff"),mask.X,mask.Y)
        glUniform2i(shader_fsq.uni("uSize"),mask.w,mask.h)

        glBindVertexArray(vao_fsq )
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0 )

       
        # Get the current zoom and center
        c = glm.vec2(mask_xpos / float(W) * 2.0 - 1.0,(H - mask_ypos) / float(H) * 2.0 - 1.0)

        if is_translating:
            curr_tra = c -  glm.vec2(tra_xstart / float(W) * 2.0 - 1.0,(H - tra_ystart) / float(H) * 2.0 - 1.0)
        

        # Apply the mask zoom and center to the current zoom and cent

        curr_zoom = curr_zoom * mask_zoom
        curr_center = (curr_center-c)*mask_zoom + c  
        tra = curr_center + curr_tra

        if not is_translating:
            curr_center = tra
            curr_tra = glm.vec2(0.0, 0.0)


        t1 = curr_zoom * 1.0 + curr_center.x + tra.x < 1.0
        t2 = curr_zoom * 1.0 + curr_center.y + tra.y < 1.0
        t3 = curr_zoom * -1.0 + curr_center.x + tra.x > -1.0
        t4 = curr_zoom * -1.0 + curr_center.y + tra.y > - 1.0

        if t1 or t2 or t3 or t4:
            curr_zoom = 1.0
            curr_center = glm.vec2(0.0, 0.0)
            curr_tra = glm.vec2(0.0, 0.0)
            tra = glm.vec2(0.0, 0.0)
           
        mask_zoom = 1.0

        # Set the zoom and center for the full screen quad shader   

        #zoom stuff
        if not show_mask:
            glUniform1f(shader_fsq.uni("uSca"), 0.0)
        else:
            glUniform1f(shader_fsq.uni("uSca"), curr_zoom )
        glUniform2f(shader_fsq.uni("uTra"), tra.x, tra.y)  


        glActiveTexture(current_unit)
        glBindTexture(GL_TEXTURE_2D, current_texture)
        glUseProgram(0)


def display(shader0, r,tb,detect,get_uvmap):
    global polyps
    global show_image
    global user_matrix
    global projection_matrix
    global view_matrix
    global user_camera
    global id_camera
    global cameras
    global texture_IMG_id
    global vao_fsq
    global project_image
    global W
    global H
    global id_comp  
    global id_node 
    global show_all_masks 
    global show_all_comps

    

    if(detect):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_uv.id_fbo)
        glDrawBuffers([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3])
        glViewport(0,0,fbo_uv.w,fbo_uv.h)
    else:
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glViewport(0,0,W,H)

    glClearColor (1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader0.program)

    # make the chunk transformation
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)
    chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix =  glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl))
    chunk_sca_matrix =  glm.scale(glm.mat4(1.0),  glm.vec3(chunk_scal))
#    chunk_matrix = glm.mul(chunk_tra_matrix,glm.mul(chunk_sca_matrix, chunk_rot_matrix))
    chunk_matrix = chunk_tra_matrix* chunk_sca_matrix* chunk_rot_matrix
    
    #camera_frame = glm.mul(chunk_matrix, (glm.transpose(glm.mat4(*cameras[id_camera].transform))))
    camera_frame = chunk_matrix * (glm.transpose(glm.mat4(*cameras[id_camera].transform)))
    camera_matrix = glm.inverse(camera_frame)

    # Initialize min_z (near) and max_z (far) before the loop

    if False and get_uvmap:  # need to update near and far, compute it the first time
        cam = cameras[id_camera]
        if not hasattr(cam, 'min_z'):
            cam.frame = camera_frame
            cam.min_z = float('inf')
            cam.max_z = float('-inf')
            for v in vertices:
                v_hom = glm.vec4(*v, 1.0)
                v_cam = camera_matrix * v_hom
                cam.min_z = min(cam.min_z, v_cam.z)
                cam.max_z = max(cam.max_z, v_cam.z)

        glUniform1f(shader0.uni("near"), cameras[id_camera].min_z)
        glUniform1f(shader0.uni("far"), cameras[id_camera].max_z)

   
    if(user_camera and not detect):
        # a view of the scene
        view_matrix = user_matrix
        projection_matrix = glm.perspective(glm.radians(45), 1.5,0.1,10)
        glUniformMatrix4fv(shader0.uni("uProj"),1,GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader0.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))
    else:
        # the view from the current id_camera
        view_matrix = camera_matrix
        maskout.current_camera_matrix = camera_matrix

    glUniformMatrix4fv(shader0.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"),user_camera)
    glUniform1i(shader0.uni("uModeProj"),project_image)

    glActiveTexture(GL_TEXTURE0)
    if(project_image):
        # texture the geometry with the current id_camera image
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniformMatrix4fv(shader0.uni("uViewCam"),1,GL_FALSE,  glm.value_ptr(camera_matrix))
    else:
        # use the texture of the mesh
        glBindTexture(GL_TEXTURE_2D, r.texture_id)

    #print(f"mat: {projection_matrix*view_matrix}, \n P(0)={projection_matrix*view_matrix*tb.matrix()*glm.vec4(0,0,0,1)}")
    #draw the geometry

    glBindVertexArray( r.vao )
    glDrawArrays(GL_TRIANGLES, 0, r.n_faces*3  )
    glBindVertexArray( 0 )

    if not detect: 
        #glDisable(GL_DEPTH_TEST)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 1.0)
        for idx, pol in enumerate(polyps):
            if show_all_comps or idx == id_comp:
                glPointSize(5)
                glColor3f(0.0, 0.0, 1.0)

                #mat = glm.mul(tb.matrix(), glm.translate(glm.mat4(1.0), glm.vec3(*pol.centroid_3D)))
                mat =  tb.matrix()* glm.translate(glm.mat4(1.0), glm.vec3(*pol.centroid_3D))

                glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(mat))
                gluSphere(quadric,0.0005,8,8)
                glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))

            # glBegin(GL_POINTS)
                #glVertex3fv(pol.centroid_3D)
                
            # glColor3f(1.0,0.0,0.0)
                #for sample in pol.samples:
                #    glVertex3fv(sample) 
            # glEnd()

                glBegin(GL_LINES)
                glColor3f(1.0, 1.0, 1.0)
                glVertex3fv(pol.centroid_3D)
                glVertex3fv(pol.tip_0)
                glColor3f(1.0, 0.0, 0.0)
                glVertex3fv(pol.centroid_3D)
                glVertex3fv(pol.tip_1)
                glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        #glEnable(GL_DEPTH_TEST)

        for idx, pol in enumerate(polyps):
            if show_all_comps or idx == id_comp:
                glBegin(GL_LINES)
                glColor3f(0.0, 1.0, 0.0)
                glVertex3fv(pol.centroid_3D)
                glVertex3fv(pol.normal_tip)
                glVertex3fv(pol.centroid_3D)
                glVertex3fv(pol.centroid_3D + (pol.centroid_3D-pol.normal_tip))
                glEnd()


    #  draw the camera frames
    if(not detect  and  user_camera):
        for i in range(0,len(cameras)):
            camera_frame = chunk_matrix * ((glm.transpose(glm.mat4(*cameras[i].transform))))
            track_mul_frame = tb.matrix()*camera_frame

            glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(track_mul_frame))
            

            glBindVertexArray(vao_frame )
            glDrawArrays(GL_LINES, 0, 6)
            glBindVertexArray( 0 )

    glUseProgram(0)

    if(user_camera == 0 and show_image ):
        #draw the image as a full screen
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(shader_fsq.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniform1i(shader_fsq.uni("uColorTex"),0)

        glBindVertexArray(vao_fsq )
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0 )

        glUseProgram(0)
        glDisable(GL_BLEND)    

    glBindFramebuffer(GL_FRAMEBUFFER,0)

    
    if False and detect and get_uvmap :
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex1)
        buf =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf)
        maskout.uv_map =np.flipud(np.frombuffer(buf, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (maskout.uv_map * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_color_{id_camera}.png")   


        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex2)
        buf = np.empty((fbo_uv.h, fbo_uv.w, 3), dtype=np.float32)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf)
        maskout.triangle_map =np.flipud(buf)
        uv_map_uint8 = (maskout.triangle_map * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_idtriangles_{id_camera}.png")    
 

    if False and get_uvmap:
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_depth)

        colatt1 =   bytearray(fbo_uv.h* fbo_uv.w* 4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, colatt1)
        colatt1 =np.flipud(np.frombuffer(colatt1, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w,1)))
        uv_map_uint8 = (colatt1 * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8.squeeze(), 'L')
        image.save(f"output_depth_{id_camera}.png")     
        

    if  False and maskout.DBG_writeout :
            
        # save the uvmap
        uv_map_uint8 = (maskout.uv_map * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_uvmap_{id_camera}.png")        
        
        #read the colorattachment
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex)

        colatt1 =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, colatt1)
        colatt1 =np.flipud(np.frombuffer(colatt1, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (colatt1 * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_color_{id_camera}.png")     
        
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex2)
        idtriangles =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, idtriangles)
        idtriangles =np.flipud(np.frombuffer(idtriangles, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (idtriangles * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_idtriangles_{id_camera}.png")     


def clicked(x,y):
    global tb 
    y = viewport[3] - y
    depth = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT).item()
    mm = np.array(view_matrix*tb.matrix(), dtype=np.float64).flatten()
    pm = np.array(projection_matrix, dtype=np.float64).flatten()
    p  =gluUnProject(x,y,depth, mm,pm, np.array(viewport, dtype=np.int32))

    p_NDC = np.array([x/float(viewport[2])*2.0-1.0,y/float(viewport[3])*2.0-1.0,-1.0+2.0*depth], dtype=np.float64)
    p_mm = glm.inverse(projection_matrix) * glm.vec4(p_NDC[0],p_NDC[1],p_NDC[2], 1.0)
    p_mm /= p_mm.w
    p_w = glm.inverse(view_matrix) * p_mm
    p_w /= p_w.w
    p = p_w
    return p, depth

def load_camera_image( id):
    global id_loaded
    global texture_IMG_id
    filename =   imgs_path +"/"+ cameras[id].label+".JPG" 
    print(f"loading {cameras[id].label}.JPG")
    glDeleteTextures(1, [texture_IMG_id])
    texture_IMG_id,_,__ = texture.load_texture(filename)
    maskout.texture_IMG_id = texture_IMG_id
    id_loaded = id

def load_mesh(filename):
    global ms
    # Load the mesh using PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.current_mesh()

    # Extract vertices, faces, and texture coordinates
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    wed_tcoord = mesh.wedge_tex_coord_matrix()
    if( mesh.has_wedge_tex_coord()):
         ms.apply_filter("compute_texcoord_transfer_wedge_to_vertex")

   
    
    dbg_tex_coords = mesh.vertex_tex_coord_matrix() if mesh.has_vertex_tex_coord() else None
    print(f"tex cord face {faces[0]}")
    print(f":  {wed_tcoord[0]}")

    print(f"dbg text coord {dbg_tex_coords[0:10]}")


#    vert_color = mesh.vertex_color_matrix()

    texture_id = -1
    if mesh.textures():
        texture_dict = mesh.textures()
        texture_name = next(iter(texture_dict.keys()))  # Get the first key    
        texture_id,w,h = texture.load_texture(texture_name)

    maskout.domain_mask = np.full((h, w, 3), 0, dtype=np.uint8)
    maskout.domain_mask_glob = np.full((h, w), -2, dtype=int)
    maskout.triangle_domain = np.full(len(faces)*3,-1,dtype=int)

    imgdata = Image.open(texture_name)
   # maskout.domain_mask =  np.flipud(np.array(imgdata, dtype=np.uint8))

    # Compute the bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    print(f"Bounding Box Min: {bbox_min}")
    print(f"Bounding Box Max: {bbox_max}")

    # Compute the diagonal length of the bounding box
    diagonal_length = ((bbox_max - bbox_min) ** 2).sum() ** 0.5
    print(f"Bounding Box Diagonal Length: {diagonal_length}")

    print(f"vertices: {len(vertices) }")
    print(f"faces: {len(faces)}")

    return vertices, faces, wed_tcoord, bbox_min,bbox_max,texture_id, w,h

def estimate_range():
    global masks_filenames
    global masks_path
    global id_camera
    global detect
    global user_camera
    global rend
    global shader0
   
    global n_masks

    print("Estimating range...")
    ranges = []
    id_mask_to_load_R   = 0
   

    n = 100 # len(masks_filenames)
    n_rest = n

    while n_rest > 0:
        start_time = time.time()
        masks_to_process = []
        for im in range(id_mask_to_load_R,len(masks_filenames)):
            mask = maskout.load_mask(masks_path,masks_filenames[im])
            if mask.C > 0.8 :
                id_cam = mask.id_camera
                if id_cam != id_camera:
                    if n_rest == n:
                        id_camera = id_cam
                        get_uvmap = True
                        display(shader0, rend,tb, detect,get_uvmap)
                    else:   
                        break
                
                masks_to_process.append(mask)    
                n_rest -= 1
                if n_rest == 0:
                    break

        id_mask_to_load_R = im + 1 
        if id_mask_to_load_R >= len(masks_filenames):
            print("No more masks to process.")
            n_rest = 0
      
        n = n_rest

        print(f"time elapsed: {time.time() - start_time:.2f} seconds")

        # Process masks in chunks of 16
        chunk_size = 128
        for i in range(0, len(masks_to_process), chunk_size):
            chunk = masks_to_process[i:min(i+chunk_size,len(masks_to_process))]         
            ranges += list(maskout.compute_range(chunk))

    # Calculate the average range   
    if False:
        if ranges:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(ranges, bins=30, color='blue', alpha=0.7)
            plt.title("Distribution of Ranges")
            plt.xlabel("Range")
            plt.ylabel("Frequency")
            # Calculate and show the 80th percentile
            p80 = np.percentile(ranges, 30)
            plt.axvline(p80, color='red', linestyle='dashed', linewidth=2, label=f' percentile: {p80:.2f}')
            plt.legend()
            plt.show()

    return np.percentile(ranges, 70)

def process_masks(n):
    global masks_filenames
    global masks_path
    global id_camera
    global detect
    global user_camera
    global rend
    global shader0
    global id_mask_to_load
    global n_masks
    global range_threshold
    

    detect = True
    user_camera = False
    get_uvmap = False

    
    
    if not hasattr(process_masks, "first"):
        process_masks.first = True # Initialize once
        get_uvmap = True
        load_camera_image(id_camera)
        display(shader0, rend,tb, detect,get_uvmap) 
        range_threshold = estimate_range()
   

    if n == 0:
        return
    
    n_rest = n

    while n_rest > 0:
        masks_to_process = []
        for im in range(id_mask_to_load,len(masks_filenames)):
            mask = maskout.load_mask(masks_path,masks_filenames[im])
            if mask.C > 0.8 :
                id_cam = mask.id_camera
                if id_cam != id_camera:
                    if n_rest == n:
                        id_camera = id_cam
                        get_uvmap = True
                        display(shader0, rend,tb, detect,get_uvmap)
                        load_camera_image(id_camera)
                    else:   
                        break

                compute_plane_slantedness(mask)
                masks_to_process.append(mask)    
                n_rest -= 1
                if n_rest == 0:
                    break

        id_mask_to_load = im + 1 
        if id_mask_to_load >= len(masks_filenames):
            print("No more masks to process.")
            return
      
        n = n_rest

        # Process masks in chunks of 16
        chunk_size = 128
        for i in range(0, len(masks_to_process), chunk_size):
            chunk = masks_to_process[i:min(i+chunk_size,len(masks_to_process))]         
            maskout.process_masks_GPU(chunk,range_threshold)
            

    n_masks = n_rest

    


def process_mask(mask):
    global detect
    global user_camera
    global masks_path
    global rend
    global shader0
    global id_camera

    if not hasattr(process_mask, "first"):
        process_mask.first = 1  # Initialize once

    detect = True
    user_camera = False
    get_uvmap = False

    # load the mask
 
    next_id_camera = -1
    for i  in   range(0,len(cameras)) :
        if   (cameras[i].label  == mask.img_name):
            next_id_camera = i        

    if next_id_camera == -1:
        print(f"ERROR Camera not found for mask {mask.img_name}!")

    if(process_mask.first ==1  or  not next_id_camera == id_camera):
        id_camera = next_id_camera
        get_uvmap = True
    
    process_mask.first = 0

    check_gl_errors()
    if get_uvmap :
        display(shader0, rend,tb, detect,get_uvmap) 
    check_gl_errors()

    get_uvmap = False

    #maskout.process_mask(mask)
    #maskout.process_mask_GPU(mask)
    maskout.process_masks_GPU([mask])
    

#    glActiveTexture(GL_TEXTURE3)
#    glBindTexture(GL_TEXTURE_2D, rend.mask_id)
#    datatoload = np.flipud(maskout.domain_mask).astype(np.uint8) 
#    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8192, 8192, 0,  GL_RGB,  GL_UNSIGNED_BYTE, datatoload)


  
    detect = False

def load_masks(masks_path):
    global masks_filenames
    masks_filenames = []
    for filename in os.listdir(masks_path):
        file_path = os.path.join(masks_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file, not a folder
            masks_filenames.append(filename)

def refresh_domain():
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, rend.mask_id)
    datatoload = np.flipud(maskout.domain_mask).astype(np.uint8) 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8192, 8192, 0,  GL_RGB,  GL_UNSIGNED_BYTE, datatoload)
    
    curr_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
    curr_vbo = glGetIntegerv(GL_ARRAY_BUFFER_BINDING)
    glBindVertexArray(rend.vao)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    
    glBufferData(GL_ARRAY_BUFFER,maskout.tri_color.nbytes, maskout.tri_color, GL_STATIC_DRAW)
    glBindVertexArray(curr_vao)
    glBindBuffer(GL_ARRAY_BUFFER, curr_vbo)





class polyp:
    def __init__(self, id_comp, id_mask, area, orientation,centroid,max_diam, min_diam, avg_col):
        self.id_comp = id_comp
        self.id_mask = id_mask  #index to the maskin allmasks
        self.area = area
        self.orientation = orientation
        self.centroid = centroid
        self.max_diam = max_diam
        self.min_diam = min_diam
        self.avg_col = avg_col


def estimate_plane(mask):
    global buf_pos
    global id_stored_pos
    global stored_pos
    global id_camera
    global detect 
    global get_uvmap
    global user_camera

  
    if id_camera != mask.id_camera:
        id_camera = mask.id_camera
        user_camera = False
        detect = True
        get_uvmap = True
        display(shader0, rend,tb, True,True)

    if not 'id_stored_pos' in globals():
        id_stored_pos = -1
       
    if id_stored_pos != mask.id_camera:    
        glBindTexture(GL_TEXTURE_2D, fbo_uv.id_tex3)
        buf_pos =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf_pos)
        stored_pos = np.flipud(np.frombuffer(buf_pos, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        id_stored_pos = mask.id_camera


    # estimate polyp plane and scale
    # Pick a random pair (row, col) in [0, mask.h) x [0, mask.w)
    samples = []
    samples_2D = []
    
    for n in range(1000):
        row = np.random.randint(0, mask.h)
        col = np.random.randint(0, mask.w)
        value_at_pixel = mask.img_data[row, col]
        if value_at_pixel > 0:
            cx = mask.X + row
            cy = mask.Y + col  
            cx_int = int(round(cx))
            cy_int = int(round(cy))
            samples_2D.append(np.array([int(round(row)), int(round(col))]))
            samples.append(stored_pos[cy_int, cx_int])
            if len(samples) >= 200:
                break       

    normal, offset,in_th = fit_plane(samples)
    return normal, offset

def compute_plane_slantedness(mask):
    global view_matrix
    normal,_  = estimate_plane(mask)
    normal_cam = view_matrix * glm.vec4(*normal, 0.0)
    normal_cam = glm.normalize(normal_cam)
    mask.ortho = abs(normal_cam.z)
    return mask.ortho > 0.8


def project_to_3D(pol):
    global buf_pos
    global id_stored_pos
    global stored_pos
    global id_camera
    global detect 
    global get_uvmap
    global user_camera

    mask = maskout.all_masks.nodes[pol.id_mask].mask
     

    if id_camera != mask.id_camera:
        id_camera = mask.id_camera
        user_camera = False
        detect = True
        get_uvmap = True
        display(shader0, rend,tb, True,True)

    if not 'id_stored_pos' in globals():
        id_stored_pos = -1
       
    if id_stored_pos != mask.id_camera:    
        glBindTexture(GL_TEXTURE_2D, fbo_uv.id_tex3)
        buf_pos =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf_pos)
        stored_pos = np.flipud(np.frombuffer(buf_pos, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        id_stored_pos = mask.id_camera


    # estimate polyp plane and scale
    # Pick a random pair (row, col) in [0, mask.h) x [0, mask.w)
    pol.samples = []
    pol.samples_2D = []
    
    min_i = max(0,mask.w/2-15)
    min_j = max(0,mask.h/2-15)
    max_i = max(0,mask.w/2+15)
    max_j = max(0,mask.h/2+15)

    for i in range(int(min_i), int(max_i)):
        for j in range(int(min_j), int(max_j)):
            if i >= 0 and i < mask.w and j >= 0 and j < mask.h:
                value_at_pixel = mask.img_data[j, i]
                if value_at_pixel > 0:
                    cx = mask.X + i
                    cy = mask.Y + j
                    cx_int = int(round(cx))
                    cy_int = int(round(cy))
                    pol.samples_2D.append(np.array([i, j]))
                    pol.samples.append(stored_pos[cy_int, cx_int])

    pol.normal, pol.offset,in_th = fit_plane(pol.samples)

    cx = mask.X + pol.centroid[0]
    cy = mask.Y + pol.centroid[1]   

    # Access the pos value at coordinates (cx, cy) in stored_pos
    # Ensure cx, cy are within bounds and are integers
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    pol.centroid_3D = stored_pos[cy_int, cx_int]
    pol.centroid_3D,_ = project_point_on_plane(pol.centroid_3D, pol.normal, pol.offset)

    # Project all sample points onto the estimated polyp plane
    scale_samples = []
    for i, sample_2D in enumerate(pol.samples_2D):
        sp,d = project_point_on_plane(pol.samples[i], pol.normal, pol.offset)
        pol.samples[i] = sp # temporary store the projected point
        if d <= in_th :
            scale = np.linalg.norm(pol.centroid_3D - sp) / np.linalg.norm(pol.centroid - pol.samples_2D[i]) 
            scale_samples.append(scale)
    
    mean_scale = np.mean(scale_samples)
    var_scale = np.var(scale_samples)
    print(f"ID {pol.id_comp} mean : {mean_scale}, Variance: {var_scale}, SNR: {mean_scale/np.sqrt(var_scale)}")

    pol.area = pol.area * mean_scale**2
    pol.confidence  = mean_scale/np.sqrt(var_scale)
    pol.max_diam = pol.max_diam * mean_scale
    pol.min_diam = pol.min_diam * mean_scale

    # Project all sample points onto the estimated polyp plane
   #pol.samples = [project_point_on_plane(np.array(s), pol.normal, pol.offset) for s in pol.samples]

    # find the principal axis
    axis = np.array([1.0, 0.0])
    c, s = np.cos(pol.orientation), np.sin(pol.orientation)
    rotation_matrix = np.array([[c, -s], [s, c]])
    principal_axis = rotation_matrix @ axis
    pol.axis_0 = principal_axis
    pol.axis_1 = np.array([-principal_axis[1], principal_axis[0]])
     
    cx = mask.X + pol.centroid[0]+pol.axis_0[0] * pol.max_diam / 3
    cy = mask.Y + pol.centroid[1]+pol.axis_0[1] * pol.max_diam / 3

    cx_int = int(round(cx))
    cy_int = int(round(cy))

    pol.tip_0 = stored_pos[cy_int, cx_int]
    pol.tip_0 ,_ = project_point_on_plane(pol.tip_0, pol.normal, pol.offset)

    pol.normal_tip = pol.centroid_3D + pol.normal * 0.01

    cx = mask.X + pol.centroid[0]+pol.axis_1[0] * pol.min_diam / 3
    cy = mask.Y + pol.centroid[1]+pol.axis_1[1] * pol.min_diam / 3

    cx_int = int(round(cx))
    cy_int = int(round(cy))

    pol.tip_1 = stored_pos[cy_int, cx_int]
    pol.tip_1,_ = project_point_on_plane(pol.tip_1, pol.normal, pol.offset)

   #scale_0 = np.linalg.norm(pol.centroid_3D - pol.tip_0)/pol.max_diam
   # scale_1 = np.linalg.norm(pol.centroid_3D - pol.tip_1)/pol.min_diam
  #  print(f"id_pol {pol.id_comp} n masks {len(maskout.all_masks.connected_components[pol.id_comp])} scale_0 {scale_0} scale_1 {scale_1}")


def fill_polyps():
    global polyps
    polyps = []

    # Find the node with the highest mask.ones in each connected component
    for comp_idx, component in enumerate(maskout.all_masks.connected_components):
        max_ones = -1
        max_idx = -1
        for idx in component:
            node = maskout.all_masks.nodes[idx]
            if node.mask.ones > max_ones:
                max_ones = node.mask.ones
                max_idx = idx
                
        component.best_node = max_idx
        node = maskout.all_masks.nodes[max_idx]
        mask = node.mask
        centroid, area, max_diam, min_diam, orientation  = metrics.compute_metrics(mask.img_data)
        curr_pol = polyp(comp_idx,max_idx, area, orientation, centroid, max_diam, min_diam,mask.avg_col)
        project_to_3D(curr_pol)
        polyps.append(curr_pol)

def compute_bounding_boxes_per_camera():
    global main_path
    global id_camera
    shader_bbox = maskout.cshader(bbox_shader_str)
    glUseProgram(shader_bbox.program)
    glUniform1i(shader_bbox.uni("uColorTexture"), 6)
    glActiveTexture(GL_TEXTURE6)
    glBindTexture(GL_TEXTURE_2D, fbo_uv.id_tex3)  # Assuming this is the texture with the color data
   

    bbox = np.zeros( 4, dtype=np.uint32) # m.w*m.h to be replaced with the max value of all the masks
    bbox[0] = 6000  # min_x
    bbox[1] = 4000  # min_y
    bbox[2] = 0     # max_x
    bbox[3] = 0     # max_y
    bbox_sbbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bbox_sbbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, bbox.nbytes, bbox, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, bbox_sbbo)
    glUseProgram(0)
    for i, cam in enumerate(cameras):
        print(f"Processing camera {i}: {cam.label}")
        # Set up the camera view
        id_camera = i
        display(shader0, rend,tb, True,True)
        glUseProgram(shader_bbox.program)

        bbox[0] = 6000  # min_x
        bbox[1] = 4000  # min_y
        bbox[2] = 0     # max_x
        bbox[3] = 0     # max_y
        glBufferData(GL_SHADER_STORAGE_BUFFER, bbox.nbytes, bbox, GL_DYNAMIC_COPY)

        glDispatchCompute(int(6000/32+1),int(4000/32+1),1)
        glUseProgram(0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bbox_sbbo)
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bbox.nbytes, GL_MAP_READ_BIT)

        data_ptr = cast(ptr, POINTER(np.ctypeslib.ctypes.c_uint32))
        bbox[:] = np.ctypeslib.as_array(data_ptr, shape=(bbox.size,))
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

        #enlarge the bbox by 10 pixels
        bbox[0] = max(10, bbox[0])-10  # min_x
        bbox[1] = max(10, bbox[1])-10  # min_y
        bbox[2] = min(6000, bbox[2] + 10)  # max_x
        bbox[3] = min(4000, bbox[3] + 10)  # max_y


        # Write bbox[0:3] to a txt file
        full_path = os.path.join(imgs_path, f"{cam.label}.txt")
        with open( full_path, "w") as f:
            f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def segment_imgs():
    global app_path
    global detect
    global user_camera
    global get_uvmap

    detect = True
    user_camera = False
    get_uvmap = False

    compute_bounding_boxes_per_camera()
    os.chdir(app_path)
    apply_yolo(imgs_path,masks_path)
    os.chdir(app_path)

def show_node(id_node):
    if(id_node > len(maskout.all_masks.nodes)-1 ):
        id_node = len(maskout.all_masks.nodes)-1
    if(id_node < 0):
        id_node = 0
    maskout.clear_domain()
    maskout.color_the_domain(maskout.all_masks.nodes[id_node].mask,maskout.all_masks.nodes[id_node].color)
    refresh_domain()

def show_comp(id_comp):
    if(id_comp > len(maskout.all_masks.connected_components)-1 ):
        id_comp = len(maskout.all_masks.connected_components)-1
    if(id_comp < 0):
        id_comp = 0
    maskout.clear_domain()
    maskout.color_connected_component(id_comp)



def export_stats():
    # Export statistics to an Excel file
    stats_file = os.path.join(main_path, "stats.xlsx")
    data = []

    # Collect statistics for each polyp
    for pol in polyps:
        data.append({
            "id_polyp": pol.id_comp,
            "mask_file": maskout.all_masks.nodes[pol.id_mask].mask.filename,
            "confidence (>2)":pol.confidence,
            "area": pol.area,
            "max_diam": pol.max_diam,
            "min_diam": pol.min_diam,
            "centroid_3D_x": getattr(pol, "centroid_3D", [None, None, None])[0],
            "centroid_3D_y": getattr(pol, "centroid_3D", [None, None, None])[1],
            "centroid_3D_z": getattr(pol, "centroid_3D", [None, None, None])[2],
            "avg_color": pol.avg_col
        })

    df = pd.DataFrame(data)
    try:
        df.to_excel(stats_file, index=False)
        print(f"Stats exported to {stats_file}")
    except Exception as e:
        print(f"Failed to write stats to {stats_file}: {e}")

def reset_display_image():
    global mask_zoom
    global mask_xpos
    global mask_ypos
    global mouseX
    global mouseY
    global curr_zoom
    global curr_center
    global is_translating   
    global tra_xstart
    global tra_ystart
    global curr_tra
    global show_mask
    global show_image


    show_mask = False
    show_image = False
    mask_zoom = 1.0
    mask_xpos = W/2
    mask_ypos = H/2
    mouseX = W/2
    mouseY = H/2
    curr_zoom   = 1.0
    curr_center = glm.vec2(0.0, 0.0)
    is_translating  = False   
    tra_xstart  = mouseX 
    tra_ystart = mouseX
    curr_tra = glm.vec2(0.0, 0.0)
    show_mask = False

def clear_redundants():
    to_remove = []
    for idx_comp, comp in enumerate(maskout.all_masks.connected_components):
       survived = maskout.all_masks.test_redundancy(idx_comp)
       if survived < 0.4:
            to_remove.append(idx_comp) 
    print (f'coomponents to remove {to_remove}')
    # Remove components indexed in to_remove from maskout.all_masks.connected_components
    #for idx in reversed(to_remove):
    #    del maskout.all_masks.connected_components[idx]
    for idx in reversed(to_remove):
        maskout.all_masks.connected_components[idx].deleted = True


def quantify_coverage(m):
    return sum(m.triangles.values())/m.ones_w
    
def count_polyps():
    global show_image
    global show_mask
    global cov_thr

    show_image = False
    show_mask = False
    reset_display_image()

    #new version..remove the mask to be removed    
    newnodes = []
    for node in maskout.all_masks.nodes:
        mask = node.mask
       # if  float(len(mask.triangles)) / mask.n_triangles  > 0.5:  
        if quantify_coverage(mask) > cov_thr:
            newnodes.append(node)
    maskout.all_masks.nodes = newnodes

    #there will be only 1 element compoent
    maskout.all_masks.count_connected_components() 
    component_sizes = {}
    for component in maskout.all_masks.connected_components:
        size = len(component)
        if size not in component_sizes:
            component_sizes[size] = 0
        component_sizes[size] += 1
    sorted_sizes = sorted(component_sizes.items(), key=lambda x: x[0], reverse=True)
    cumulative_sum = 0
    for size, count in sorted_sizes:
        cumulative_sum += count
        print(f"Size: {size}, Count: {count}, Cumulative Sum: {cumulative_sum}")

    fill_polyps()
    # clear_redundants()
    refresh_domain()

def main():


    glm.silence(4)
    global W
    global H
    W = 1200
    H = 800
    global masks_filenames
    global tb
    global  sensor
    global vertices 
    global chunk_rot
    global chunk_transl
    global chunk_scal
    global vao_frame
    global shader_fsq
    global texture_IMG_id
    global show_image
    global vao_fsq
    global id_loaded
    global project_image
    global domain_mask 
    global detect
    global masks_path
    global rend
    global shader0
    global user_matrix
    global id_mask_to_load
    global n_masks
  #  global show_metrics
    global polyps
    global app_path
    global main_path
    global imgs_path
    global metashape_file
    global show_all_masks
    global show_all_comps
    global id_comp 
    global id_node 
    global range_threshold
    global mask_zoom
    global mask_xpos
    global mask_ypos
    global mouseX
    global mouseY
    global curr_zoom
    global curr_center
    global is_translating   
    global tra_xstart
    global tra_ystart
    global curr_tra
    global show_mask
    global quadric
    global cov_thr

    show_mask = False

    is_translating = False



    with open("last.txt", "r") as f:
        content = f.read()
        print("Raw content:", repr(content))

        lines = content.splitlines()
        print("lines:", lines)
        if len(lines) >= 4:
            main_path = lines[0]
            imgs_path = lines[1]
            masks_path = lines[2]
            mesh_name = lines[3]
            metashape_file = lines[4]
        else:
            print("last.txt does not contain enough lines.")




    if len(sys.argv) > 1:
        main_path = sys.argv[1]
        imgs_path = sys.argv[2]
        masks_path = sys.argv[3]
        mesh_name = sys.argv[4]

    print(f"params: {sys.argv}")

    polyps = []
    id_mask_to_load = 0
    id_loaded = -1
    show_image = False
    project_image = False

    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    pygame.display.set_caption("Polyp Detector")
  
    max_ssbo_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
    print(f"Max SSBO size: {max_ssbo_size / (1024*1024):.2f} MB")

    # Initialize ImGui
    imgui.create_context()
    imgui_renderer = PygameRenderer()

    # Set ImGui's display size to match the window size
    imgui.get_io().display_size = (W,H)  # Ensure valid display size

    quadric = gluNewQuadric()

    tb = trackball.Trackball()
    tb.reset()
    glClearColor(1, 1,1, 0.0)
    glEnable(GL_DEPTH_TEST)
    
    app_path = os.path.dirname(os.path.abspath(__file__))
    print(f"App path: {app_path}")

 

 

    os.chdir(main_path)
    vertices, faces, wed_tcoords, bmin,bmax,texture_id,texture_w,texture_h  = load_mesh(mesh_name) #load_mesh("Sample3_2024-09_Pht-TextMESH.ply")
    

    load_masks(masks_path)
#    mask = maskout.load_mask(main_path+"/"+masks_path,"IMG_0038_02834_01635_0.91609.png")


    sensor  = metashape_loader.load_sensor_from_xml(metashape_file)
    global cameras 
    cameras,chunk_rot,chunk_transl,chunk_scal = metashape_loader.load_cameras_from_xml(metashape_file)
    maskout.cameras = cameras

    chunk_rot = np.array(chunk_rot)
    chunk_transl = np.array(chunk_transl)

    
    rend = renderable(vao=None,n_verts=len(vertices),n_faces=len(faces),texture_id=texture_id,mask_id=texture.create_texture(texture_w,texture_h))


    global shader0

    shader0     = shader(vertex_shader, fragment_shader)
    shader_fsq  = shader(vertex_shader_fsq, fragment_shader_fsq)
   
    glUseProgram(shader0.program)
    glUniform1i(shader0.uni("uMasks"),3)
    glUniform1i(shader0.uni("resolution_width"),sensor.resolution["width"])
    glUniform1i(shader0.uni("resolution_height"),sensor.resolution["height"])
    glUniform1f(shader0.uni("f" ) ,sensor. calibration["f"]) 
    glUniform1f(shader0.uni("cx"),sensor.calibration["cx"])
    glUniform1f(shader0.uni("cy"),-sensor.calibration["cy"])
    glUniform1f(shader0.uni("k1"),sensor.calibration["k1"])
    glUniform1f(shader0.uni("k2"),sensor.calibration["k2"])
    glUniform1f(shader0.uni("k3"),sensor.calibration["k3"])
    glUniform1f(shader0.uni("p1"),sensor.calibration["p1"])
    glUniform1f(shader0.uni("p2"),sensor.calibration["p2"])
    glUseProgram(0)

    check_gl_errors()

    global fbo_uv
    fbo_uv  = fbo.fbo(sensor.resolution["width"],sensor.resolution["height"])
    maskout.fbo_uv = fbo_uv


    # for each pixel, to which triangle it belongs
    maskout.triangle_map_texture = fbo_uv.id_tex2

    # for each triangle, the index to a node/mask that covers it
   
    #w_ntri =int(np.ceil(len(faces)/2048))
    maskout.node_pointer  = texture.create_texture(2048,2048,"int32")

    # for each triangle, how many samples of the current mask fall onto it
    maskout.coverage  = texture.create_texture(2048,2048,"int32")
    maskout.adjacents = texture.create_texture(2048,2048,"int32")


    maskout.vertices = vertices
    maskout.faces = faces

    maskout.setup_cshader()
   
    maskout.triangles_nodes = [[] for _ in range(len(faces))] 
    #output_mask = texture.create_texture(sensor.resolution["width"],sensor.resolution["height"])

    #faces = np.array(faces, dtype=np.uint32).flatten()
    print(f"vertices: {len(vertices) }")
    print(f"faces: {len(faces)}")

    #print(faces)
    vertex_array_object = create_buffers(vertices,wed_tcoords,faces,shader0)
    rend.vao = vertex_array_object
    vao_frame = create_buffers_frame()
    vao_fsq = create_buffers_fsq()

    global viewport
    clock = pygame.time.Clock()
    viewport =[0,0,W,H]
    center = (bmin+bmax)/2.0
    eye = center + glm.vec3(2,0,0)
    user_matrix = glm.lookAt(glm.vec3(eye),glm.vec3(center), glm.vec3(0,0,1))
    projection_matrix = glm.perspective(glm.radians(45),1.5,0.1,10)


    global id_camera
    global user_camera
    id_camera   = 0

    user_camera = 1
    detect = False
    #go_process_masks = False
    go_process_all_masks = False
    i_toload = 0
    n_masks = len(masks_filenames  )
    id_comp = 0
    id_node = 0
    texture_IMG_id = 0
    th = 10
    prevtime = 0
  #  show_metrics = False
    show_all_masks = True
    show_all_comps  = True
    range_threshold = 100
    mask_zoom = 1.0
    mask_xpos = W/2
    mask_ypos = H/2
    curr_zoom = 1.0
    curr_center = glm.vec2(0,0)
    tra_xstart = mask_xpos
    tra_ystart = mask_ypos
    curr_tra = glm.vec2(0,0)
    cov_thr = 0.6

    while True:    
        time_delta = clock.tick(60)/1000.0 
        for event in pygame.event.get():
            imgui_renderer.process_event(event)
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                return  
            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                if show_image and is_translating:
                    mask_xpos = mouseX
                    mask_ypos = mouseY
                else:    
                    tb.mouse_move(projection_matrix, user_matrix, mouseX, mouseY)

            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if show_image:
                     mask_zoom = 1.1 if yoffset > 0 else 0.97
                     if yoffset > 0 :
                        mask_xpos = mouseX 
                        mask_ypos = mouseY
                else:
                    tb.mouse_scroll(xoffset, yoffset)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button not in (4, 5):
                    mouseX, mouseY = event.pos
                    keys = pygame.key.get_pressed()  # Get the state of all keys
                    if show_image:
                        is_translating = True
                        tra_xstart = mouseX
                        tra_ystart = mouseY
                        mask_xpos =  mouseX
                        mask_ypos =  mouseY
                    else:
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  
                            cp,depth = clicked(mouseX,mouseY)
                            if depth < 0.99:
                                tb.reset_center(cp)               
                        else:
                            tb.mouse_press(projection_matrix, user_matrix, mouseX, mouseY)
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY  = event.pos
                if event.button == 1:  # Left mouse button
                    if show_image:
                        is_translating = False
                    else:
                        tb.mouse_release()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    user_camera = 1 - user_camera




        # Start ImGui frame
        imgui.new_frame()

        if imgui.begin_main_menu_bar().opened:

            # first menu dropdown
            if imgui.begin_menu('Actions', True).opened:

                changed_n_masks, n_masks = imgui.input_int("n masks to process", n_masks,step=10)

                imgui.text("________________RUN ___________________")
                if(imgui.button("Segment images")):
                    segment_imgs()
                    load_masks(masks_path)

                if(imgui.button("Find polyps")):
                    reset_display_image()
                    if(i_toload < len(masks_filenames)-1 and n_masks > 0 ):
                        process_masks(n_masks)
                    count_polyps()
                
                
                if(imgui.button("export stats")):
                    export_stats()

                imgui.dummy(0.0, 20.0)
                imgui.text("______________ Visualize process ____________")
               
                if(imgui.button("project masks")):
                    reset_display_image()
                    if(i_toload < len(masks_filenames)-1 and n_masks > 0 ):
                        process_masks(n_masks)
                        refresh_domain()

                if(imgui.button("count polyps")):
                    count_polyps()
                    maskout.clear_domain()
                    for comp in maskout.all_masks.connected_components:
                        maskout.color_the_domain(maskout.all_masks.nodes[comp.best_node].mask, comp.v_color)



                # Integer input field in the dropdown menu
                changed_id, id_camera = imgui.input_int("n camera", id_camera)
                if changed_id:
                    id_camera = max(0, min(id_camera, len(cameras)-1))  # Ensure id_camera is within bounds
                if(changed_id and (show_image or project_image)):
                    load_camera_image(id_camera)
                id_camera = max(0, min(id_camera, len(cameras)-1)) 
                
                imgui.text_ansi(f"Curr camera {cameras[id_camera].label}")
                changed, user_camera = imgui.checkbox("free point of view", user_camera)
                #changed_im, show_image = imgui.checkbox("show image", show_image)
               # changed_im, project_image = imgui.checkbox("Project image", project_image)
                if(id_loaded != id_camera):
                     load_camera_image(id_camera)

                
                changed_i_node, id_node = imgui.input_int( "id shown  mask", id_node,step=1)
                # Add a small color box for the current mask color
                # Show color button only if avg_col attribute exists
               

                changed_cov_thr, cov_thr = imgui.input_float("thr cov ",cov_thr)
                if changed_cov_thr:
                    changed_i_node = True

                if  id_node < len(maskout.all_masks.nodes)-1 :   
                    if hasattr(maskout.all_masks.nodes[id_node].mask, "avg_col"):
                        color = maskout.all_masks.nodes[id_node].mask.avg_col
                        imgui.same_line()
                        imgui.color_button("##mask_color", *color, 1.0, imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP, 16, 16)


                if(changed_i_node):
                    if not changed_cov_thr:
                        show_all_masks = False
                    if(id_node > len(maskout.all_masks.nodes)-1 ):
                        id_node = len(maskout.all_masks.nodes)-1
                    if(id_node < 0):
                        id_node = 0
    
                    if not show_image:    
                        print(f"mask cov {quantify_coverage(maskout.all_masks.nodes[id_node].mask) } id cam {maskout.all_masks.nodes[id_node].mask.id_camera} ")
                        maskout.clear_domain()
                        maskout.color_the_domain(maskout.all_masks.nodes[id_node].mask,maskout.all_masks.nodes[id_node].color,cov_thr)
                        refresh_domain()
                    if show_mask:
                        id_camera = maskout.all_masks.nodes[id_node].mask.id_camera

                changed_show_mask, show_mask = imgui.checkbox("show mask on the original image 2D", getattr(maskout, "show mask", show_mask))
                if changed_show_mask:
                    show_image = show_mask
                    if show_mask:
                        id_camera = maskout.all_masks.nodes[id_node].mask.id_camera


                changed_show_all_masks, show_all_masks = imgui.checkbox("show all segmented  masks in 3D", getattr(maskout, "show_all_masks", show_all_masks))
                if changed_cov_thr:
                    changed_show_all_masks = True
                if changed_show_all_masks:
                        maskout.clear_domain()
                        if show_all_masks:
                            for node in maskout.all_masks.nodes:
                                maskout.color_the_domain(node.mask, node.color,cov_thr)
                            refresh_domain()
                        else:
                            show_node(id_node)
                        
                        
               #text = f"z depth" 
                #if( len(maskout.all_masks.nodes) > 0): 
               #      text= f"bbox diagonal {maskout.all_masks.nodes[id_node].mask.bbox.dim_z()}"    
                #imgui.text_ansi(text)

                
                 

                imgui.text_ansi(f"counted polyps: {len(maskout.all_masks.connected_components)}")

                changed_i_comp, id_comp = imgui.input_int( "id shown comp", id_comp,step=1)
                if(changed_i_comp):
                    show_all_comps = False
                    if(id_comp > len(maskout.all_masks.connected_components)-1 ):
                        id_comp = len(maskout.all_masks.connected_components)-1
                    if(id_comp < 0):
                        id_comp = 0
                    print(f"show comp {id_comp}: best {maskout.all_masks.connected_components[id_comp].best_node} nodes: {maskout.all_masks.connected_components[id_comp]}")

                    maskout.clear_domain()
                    #maskout.color_connected_component(id_comp)
                    comp = maskout.all_masks.connected_components[id_comp]
                    maskout.color_the_domain(maskout.all_masks.nodes[comp.best_node].mask, comp.v_color,cov_thr)

                    #first_mask = maskout.all_masks.connected_components[id_comp][0]
                    #metrics.compute_metrics(maskout.all_masks.nodes[first_mask].mask.img_data)
                    refresh_domain()
                
                if len(maskout.all_masks.connected_components) > 0 :
                    imgui.text_ansi(f"node: {maskout.all_masks.connected_components[id_comp].best_node}")

                changed_show_all_comps, show_all_comps = imgui.checkbox("show all polyps", getattr(maskout, "show_all_comps", show_all_comps))
                if changed_show_all_comps:
                        maskout.clear_domain()
                        if show_all_comps:
                            for comp in maskout.all_masks.connected_components:
                                if hasattr(comp, "deleted") :
                                    maskout.color_the_domain(maskout.all_masks.nodes[comp.best_node].mask, [0,0,0])
                                else:
                                    maskout.color_the_domain(maskout.all_masks.nodes[comp.best_node].mask, comp.v_color)
                                #for inode in comp:
                                #    nod = maskout.all_masks.nodes[inode]
                                #    maskout.color_the_domain(nod.mask, comp.v_color)
                        else:
                            show_comp(id_comp)
                        refresh_domain()
               # if imgui.button("refit polyp"):
               #     if len(polyps) > 0 and id_comp < len(polyps):
               #         project_to_3D(polyps[id_comp])
                
               # if imgui.button("verify"):
               #     print(f'surv {maskout.all_masks.test_redundancy(id_comp) }')   
                    

               # changed_show_metrics, show_metrics = imgui.checkbox("show metrics", show_metrics)
                #text = f"n nodes" 
                #if( len(maskout.all_masks.connected_components) > 0): 
                #     text= f"n nodes {len(maskout.all_masks.connected_components[id_comp])}"    
                #imgui.text_ansi(text)


                
                imgui.end_menu()
            

                  
        imgui.end_main_menu_bar()
            
        # if(go_process_masks and i_toload < len(masks_filenames)-1 and n_masks > 0 ):
        #     mask = maskout.load_mask(masks_path, masks_filenames[i_toload])
        #     i_toload += 1
        #     if(mask.C > 0.8):
        #         n_masks -=1
        #         print(f" time for the rest : {time.time()-prevtime}")
        #         start_time = time.time()
        #         
        #         process_mask(mask)
        #         print(f" time for process mask: {time.time()-start_time}")
        #         prevtime = time.time()
        #         if(n_masks == 0):
        #             refresh_domain()
        # else:
        #    if(n_masks==0):
        #     go_process_masks = False
        
        if(go_process_all_masks and i_toload < len(masks_filenames)-1 and n_masks > 0 ):
            process_masks(n_masks)
            refresh_domain()
            go_process_all_masks = False
        

        check_gl_errors()
        if show_image:
            display_image()
        else:
            display(shader0, rend,tb, False,False) 

        check_gl_errors()


        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())

        # Draw the UI elements
        pygame.display.flip()

        clock = pygame.time.Clock()


if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()