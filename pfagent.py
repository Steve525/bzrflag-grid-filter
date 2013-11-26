#!/usr/bin/python -tt

# A Potential Field-following agent that uses a PD controller to minimize
# the error between the direction of the field and its own movement.

#################################################################
# Seth Stewart and Stephen Clarkson
# October 2013
#
# After starting the bzrflag server, this is one way to start
# this code:
# python agent0.py [hostname] [port]
#
# Often this translates to something like the following (with the
# port name being printed out by the bzrflag server):
# python agent0.py localhost 49857
#################################################################

import sys
import math
import time
import numpy
from numpy import ones
import random
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import zeros

from bzrc import BZRC, Command

class Agent(object):
    """Class handles all command and control logic for a teams tanks."""

    def __init__(self, bzrc):
        self.bzrc = bzrc
        self.constants = self.bzrc.get_constants()
        self.commands = []
        self.mytankdata = []
        mytanks = self.bzrc.get_mytanks()
        for tank in mytanks:
            self.mytankdata.append((0.0, 0.0)) # push initial speed_error and angle_error onto list for each tank
        for item in self.constants:
            print item
        print self.constants['worldsize']
        # TODO: Move these two variables into the Tank class (in bzrc.py)!
        last_angle_error = 0 # can tweak, used only as the initial value
        last_speed_error = 0 # can tweak
        
        # FROBBING CENTRAL:
        self.k_pa = 1 # angular velocity constant
        self.k_da = 1 # angular velocity derivative constant
        self.k_ps = 1 # speed proportional control constant
        self.k_ds = 1 # speed proportional derivative control constant
        # Belief grid
        self.prior = 0.75
        self.worldsize = int(self.constants['worldsize'])
        self.belief = self.prior * numpy.ones((self.worldsize, self.worldsize))
        self.truehit = 0.97
        self.falsepositive = 0.9
        self.max_dist_to_obstacle = 90
        self.belief_threshold = 0.95
        
        # Coarse grid of beliefs for efficiently exploring the map
        self.coarse_size = 32
        self.coarse_elements = 16 # Number of random samples to average within coarse grid block
        self.coarse_grid = numpy.ones((self.coarse_size, self.coarse_size))
        
    def tick(self, time_diff):
        """Some time has passed; decide what to do next."""
        mytanks, othertanks, flags, shots = self.bzrc.get_lots_o_stuff()
        self.mytanks = mytanks
        self.othertanks = othertanks
        self.flags = flags
        self.shots = shots
        self.enemies = [tank for tank in othertanks if tank.color !=
                        self.constants['team']]

        self.commands = []

        #for tank in mytanks:
        #    self.do_move(tank)

        # Use only two tanks for demo
        # IMPORTANT: Using lots of tanks will slow them down to the point
        # of making their PD controllers nearly useless!
        self.update_coarse_grid()
        active_tank_count = len(mytanks)
        for i in range(0, active_tank_count-1):
            mytanks = self.bzrc.get_mytanks()
            tank = mytanks[i]
            if tank.status != 'alive':
                continue
            self.do_update_beliefs(tank)
            self.do_move(tank)
        results = self.bzrc.do_commands(self.commands)

    def update_coarse_grid(self):
        for x in range (0, self.coarse_size - 1):
            for y in range (0, self.coarse_size -1):
                average_belief = 0
                for i in range (0, self.coarse_elements - 1):
                    bx = random.randint(x * self.worldsize / self.coarse_size, (x + 1) * self.worldsize / self.coarse_size - 1)
                    by = random.randint(y * self.worldsize / self.coarse_size, (y + 1) * self.worldsize / self.coarse_size - 1)
                    average_belief += self.belief[bx][by]
                average_belief /= self.coarse_elements
                self.coarse_grid[x][y] = average_belief

    def do_update_beliefs(self, tank):
        if tank.status != 'alive':
            return
        
        pos, grid = self.bzrc.get_occgrid(tank.index)
        # Apply noise to the grid
        
        x_size = len(grid) - 1
        y_size = len(grid[0]) - 1
        '''
        for x in range(0, x_size):
            for y in range(0, y_size):
                if grid[x][y] == 1:
                    if random.random() > self.truehit:
                        grid[x][y] = 1
                    else:
                        grid[x][y] = 0
                else:
                    if random.random() > self.falsepositive:
                        grid[x][y] = 1
                    else:
                        grid[x][y] = 0
        '''
        # Update beliefs
        for x in range(0, x_size):
            bx = 400 + x + pos[0]
            for y in range(0, y_size):
                by = 400 + y + pos[1]
                if grid[x][y] == 1:
                    bel_occ = self.truehit * self.belief[by][bx]
                    bel_unocc = self.falsepositive * (1-self.belief[by][bx])
                    self.belief[by][bx] = bel_occ / (bel_occ + bel_unocc)
                else:
                    bel_occ = (1 - self.truehit) * self.belief[by][bx]
                    bel_unocc = (1 - self.falsepositive) * (1-self.belief[by][bx])
                    self.belief[by][bx] = bel_occ / (bel_occ + bel_unocc)
                if self.belief[by][bx] > self.belief_threshold:
                    self.belief[by][bx] = 1
       
        #for y in range(y_size,0,-1):
            #for x in range(0,x_size):
                #bx = 400 + x + pos[0]
                #by = 400 + y + pos[1]
                #print bx, by, pos[0], pos[1]
                '''
                if self.belief[bx][by] == 1.0:
                    sys.stdout.write('#'); #{0:.1f} '.format(float(self.belief[bx][by])))
                elif self.belief[bx][by] > 0.9:
                    sys.stdout.write('9');
                elif self.belief[bx][by] > 0.8:
                    sys.stdout.write('8');
                elif self.belief[bx][by] > 0.7:
                    sys.stdout.write('7');
                elif self.belief[bx][by] > 0.6:
                    sys.stdout.write('6');
                elif self.belief[bx][by] > 0.5:
                    sys.stdout.write('5');
                elif self.belief[bx][by] > 0.4:
                    sys.stdout.write('4');
                elif self.belief[bx][by] > 0.3:
                    sys.stdout.write('3');
                elif self.belief[bx][by] > 0.2:
                    sys.stdout.write('2')
                elif self.belief[bx][by] > 0.1:
                    sys.stdout.write('1')
                else:
                    sys.stdout.write('-')
                '''
                #print '|'
                #print self.belief[x + pos[0]][y + pos[1]]
            #sys.stdout.write('|\n')
        #print '\n'
        #print '================================================'
        self.update_grid(self.belief)

    def do_move(self, tank):
        """Compute and follow the potential field vector"""
        print(self.get_potential_field_vector(tank))
        v, theta = self.get_potential_field_vector(tank)
        self.commands.append(self.pd_controller_move(tank, v, theta))

    def move_to_position(self, tank, target_x, target_y):
        """Set command to move to given coordinates."""
        target_angle = math.atan2(target_y - tank.y,
                                  target_x - tank.x)
        relative_angle = self.normalize_angle(target_angle - tank.angle)
        command = Command(tank.index, 1, 2 * relative_angle, True)
        self.commands.append(command)

    def normalize_angle(self, angle):
        """Make any angle be between +/- pi."""
        angle -= 2 * math.pi * int (angle / (2 * math.pi))
        if angle <= -math.pi:
            angle += 2 * math.pi
        elif angle > math.pi:
            angle -= 2 * math.pi
        return angle

    ############################################################

    # Compute the potential field at the tank's position by summing all fields
    # that influence it. Outputs a vector (target_speed, target_angle)
    def get_potential_field_vector(self, tank):        
        delta_x = 0
        delta_y = 0
        
        # Exploration field
        exploration_force = 25
        attractive_multiplier = 0
        dist = 0
        angle = 0
        for x in range (0, self.coarse_size - 1):
            for y in range (0, self.coarse_size - 1):
                worldx = (x + 0.5) * self.worldsize / self.coarse_size - 400
                worldy = (y + 0.5) * self.worldsize / self.coarse_size - 400
                # Choose the most attractive direction and go there
                this_dist = math.sqrt((worldx - tank.x)**2 + (worldy - tank.y)**2)
                this_attractive_multiplier = (1 - abs(self.prior - self.coarse_grid[x][y])) * (-self.worldsize * math.sqrt(2) + this_dist)
                # idea: be attracted to immediate benefits AND have a long-term destination (distant but large objective)
                if self.coarse_grid[x][y] < (self.prior / 4):
                    this_attractive_multiplier = -.5 # Repel previously explored space
                if self.coarse_grid[x][y] > (self.prior + 1) / 2:
                    this_attractive_multiplier = -1 # Repel obstacles strongly
                if attractive_multiplier < 0 and dist > self.worldsize / 8:
                    delta_x += this_attractive_multiplier * exploration_force * math.cos(angle) * (self.worldsize * 2 - this_dist) / (self.worldsize * 2)
                    delta_y += this_attractive_multiplier * exploration_force * math.sin(angle) * (self.worldsize * 2 - this_dist) / (self.worldsize * 2)
                if this_attractive_multiplier > attractive_multiplier:
                    attractive_multiplier = this_attractive_multiplier
                    dist = math.sqrt((worldx - tank.x)**2 + (worldy - tank.y)**2)
                    angle = math.atan2((worldy - tank.y), (worldx - tank.x))
                
        delta_x += attractive_multiplier * exploration_force * math.cos(angle) * (self.worldsize * 2 - dist) / (self.worldsize * 2)
        delta_y += attractive_multiplier * exploration_force * math.sin(angle) * (self.worldsize * 2 - dist) / (self.worldsize * 2)
        
        # Repel ally tanks
        repulsive_force = 50
        for ally in self.mytanks:
            dist = math.sqrt((ally.x - tank.x)**2 + (ally.y - tank.y)**2)
            angle = math.atan2((ally.y - tank.y), (ally.x - tank.x))
            if dist < self.max_dist_to_obstacle:
                delta_x += -repulsive_force * (self.max_dist_to_obstacle*2 - dist) * math.cos(angle) / (self.max_dist_to_obstacle*2)
                delta_y += -repulsive_force * (self.max_dist_to_obstacle*2 - dist) * math.sin(angle) / (self.max_dist_to_obstacle*2)
        
        '''
        # Attractive field  
        attractive_force = 25
        goal_found = False
        goal_x = 0
        goal_y = 0
        #print ("Tank has flag? ")
        #print (tank.flag)
        if tank.flag == '-':
            goal = self.get_closest_flag(tank)
            if goal is not None:
                goal_found = True
                goal_x = goal.x
                goal_y = goal.y
                #print ("Closest flag: ")
                #print(goal.color)
        if not goal_found:
            bases = self.bzrc.get_bases()
            for base in bases:
                if base.color == self.constants['team']:
                    goal_x = (base.corner1_x + base.corner2_x + base.corner3_x + base.corner4_x) /4
                    goal_y = (base.corner1_y + base.corner2_y + base.corner3_y + base.corner4_y) /4
        dist = math.sqrt((goal_x - tank.x)**2 + (goal_y - tank.y)**2)
        angle = math.atan2((goal_y - tank.y), (goal_x - tank.x))
        
        # Bound the influence of the goal to 1 * attractive_force
        if dist > self.max_dist_to_obstacle:
            delta_x += attractive_force * math.cos(angle)
            delta_y += attractive_force * math.sin(angle)
        else:
            delta_x += attractive_force * dist * math.cos(angle) / self.max_dist_to_obstacle
            delta_y += attractive_force * dist * math.sin(angle) / self.max_dist_to_obstacle
        print("Attractive force: ", math.sqrt(delta_x**2 + delta_y**2))
        '''
        '''    
        # Repulsive fields
        repulsive_force = 25
        relative_corner_influence = 0.75
        for obstacle in self.bzrc.get_obstacles():
            center_x = 0 
            center_y = 0
            for point in obstacle:
                center_x = center_x + point[0]
                center_y = center_y + point[1]
                # Include the corner point
                dist = math.sqrt((point[0] - tank.x)**2 + (point[1] - tank.y)**2)
                angle = math.atan2((point[1] - tank.y), (point[0] - tank.x))
                if dist <= self.max_dist_to_obstacle * relative_corner_influence: # corner points have less influence than centroids
                    delta_x += -repulsive_force * (relative_corner_influence * self.max_dist_to_obstacle - dist)**1 * math.cos(angle) / (relative_corner_influence * self.max_dist_to_obstacle)
                    delta_y += -repulsive_force * (relative_corner_influence * self.max_dist_to_obstacle - dist)**1 * math.sin(angle) / (relative_corner_influence * self.max_dist_to_obstacle)

            center_x = center_x / len(obstacle)
            center_y = center_y / len(obstacle)
            #print ("Centroid: ", center_x, ", ", center_y)
            dist = math.sqrt((center_x - tank.x)**2 + (center_y - tank.y)**2)
            angle = math.atan2((center_y - tank.y), (center_x - tank.x))
            if dist < self.max_dist_to_obstacle:
                delta_x += -repulsive_force * (self.max_dist_to_obstacle - dist) * math.cos(angle) / self.max_dist_to_obstacle
                delta_y += -repulsive_force * (self.max_dist_to_obstacle - dist) * math.sin(angle) / self.max_dist_to_obstacle

        # Tangential fields
        tangential_force = 25
        for obstacle in self.bzrc.get_obstacles():
            center_x = 0 
            center_y = 0
            for point in obstacle:
                center_x = center_x + point[0]
                center_y = center_y + point[1]
                dist = math.sqrt((point[0] - tank.x)**2 + (point[1] - tank.y)**2)
                angle = math.atan2((point[1] - tank.y), (point[0] - tank.x))
                angle += math.pi / 2
                if dist < self.max_dist_to_obstacle * relative_corner_influence:
                    delta_x += -tangential_force * dist * math.cos(angle) / (self.max_dist_to_obstacle * relative_corner_influence)
                    delta_y += -tangential_force * dist * math.sin(angle) / (self.max_dist_to_obstacle * relative_corner_influence)
            
            center_x = center_x / len(obstacle)
            center_y = center_y / len(obstacle)
            
            dist = math.sqrt((center_x - tank.x)**2 + (center_y - tank.y)**2)
            angle = math.atan2((center_y - tank.y), (center_x - tank.x))
            angle += math.pi / 2
            if dist < self.max_dist_to_obstacle:
                delta_x += -tangential_force * math.cos(angle)
                delta_y += -tangential_force * math.sin(angle)
        '''     
        # Compute final vector
        v = min(math.sqrt(delta_x**2 + delta_y**2), float(self.constants['tankspeed']))
        theta = math.atan2(delta_y, delta_x)
        relative_theta = self.normalize_angle(theta)
        return (v, relative_theta) # v is goal vector of velocity and relative_theta is goal angle
    
    # Perform the actual update of the tank's controls using the target speed
    # and target angle. Uses a PD controller to minimize the error.
    def pd_controller_move(self, tank, target_speed, target_angle):
        tank_speed = math.sqrt(tank.vx**2 + tank.vy**2)

        # TODO: We cannot store the last angle and speed errors in the Agent object, since it processes all tanks.
        # These should be fields in the tank class and be initialized there (tank.last_angle_error, tank.last_speed_error)

        last_speed_error, last_angle_error = self.mytankdata[tank.index]

        angle_error = target_angle - tank.angle
        angle_error = self.normalize_angle(angle_error)
        delta_angle_error = angle_error - last_angle_error # for the derivative portion of the controller
        delta_angle_error = self.normalize_angle(delta_angle_error)
        print ("Angle error: ", angle_error)
        print ("Change in angle error: ", delta_angle_error)
        last_angle_error = angle_error # update the tank's last angle error so we can computer its derivative on our next cycle
        new_angvel = self.k_pa * angle_error + self.k_da * delta_angle_error # determine the new angular velocity
        
        speed_error = target_speed - tank_speed
        delta_speed_error = speed_error - last_speed_error
        print ("Speed error: ", speed_error)
        print ("Change in speed error: ", delta_speed_error)
        last_speed_error = speed_error # update our last speed error so we can computer its derivative on our next cycle
        new_speed = self.k_ps * speed_error + self.k_ds * delta_speed_error
        
        self.mytankdata[tank.index] = (last_speed_error, last_angle_error)
        
        shoot = False #(Is there an enemy tank in front of us? Can we avoid shooting our own?)
        # TODO: shoot periodically using the simple metric, closest tank at angle theta is enemy tank?
        if abs(speed_error) > 1: # There may be something blocking the tank
            shoot = True
        
        narrow_angle = math.pi / 8
        min_shoot_dist = 100
        for enemy in self.enemies:
            if enemy.status != 'alive':
                continue
            dist = math.sqrt((enemy.x - tank.x)**2 + (enemy.y - tank.y)**2)
            if abs(math.atan2(enemy.x - tank.x, enemy.y - tank.y)) < narrow_angle and dist < min_shoot_dist:
                shoot = True

        #DEBUG: Comment out if you want to test shooting
        #shoot = False

        # to switch to velocity-based tank speed, use only the parameter new_speed.
        # to use an acceleration-based tank speed, use new_speed + tank_speed
        new_angvel = self.normalize_angle(new_angvel)
        command = Command(tank.index, new_speed + tank_speed, new_angvel, shoot)
        return command
    
    # Return the closest enemy flag
    def get_closest_flag(self, tank):
        closest_flag = None
        best_dist = 2 * float(self.constants['worldsize'])
        flags = self.bzrc.get_flags()
        for flag in flags:
            # what about flags that are already captured?
            # what about current team's flag
            if flag.color == self.constants['team']:
                continue
            if flag.poss_color == self.constants['team']:
                continue
            dist = math.sqrt((flag.x - tank.x)**2 + (flag.y - tank.y)**2)
            if dist < best_dist:
                best_dist = dist
                closest_flag = flag
        if closest_flag is None:
            #print("There is no closest flag!")
            a = 1
        else:
            return closest_flag
     #########################################################

    def draw_grid(self):
        # This assumes you are using a numpy array for your grid
        width, height = grid.shape
        glRasterPos2f(-1, -1)
        glDrawPixels(width, height, GL_LUMINANCE, GL_FLOAT, grid)
        glFlush()
        glutSwapBuffers()
        glutPostRedisplay()

    def update_grid(self, new_grid):
        global grid
        grid = new_grid



    def init_window(self, width, height):
        global window
        global grid
        grid = zeros((width, height))
        glutInit(())
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("Grid filter")
        glutDisplayFunc(self.draw_grid)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        #glutMainLoop()

def main():
    # Process CLI arguments.
    try:
        execname, host, port = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print >>sys.stderr, '%s: incorrect number of arguments' % execname
        print >>sys.stderr, 'usage: %s hostname port' % sys.argv[0]
        sys.exit(-1)

    # Connect.
    #bzrc = BZRC(host, int(port), debug=True)
    bzrc = BZRC(host, int(port))

    agent = Agent(bzrc)

    prev_time = time.time()
    agent.init_window(int(800),int(800))
    # Run the agent
    try:
        while True:
            time_diff = time.time() - prev_time
            agent.tick(time_diff)
            glutMainLoopEvent()
    except KeyboardInterrupt:
        print "Exiting due to keyboard interrupt."
        bzrc.close()


if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
