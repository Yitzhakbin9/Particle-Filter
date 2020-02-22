import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
import random
import time
import math


np.random.seed(14)

@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float


@dataclass
class LandMark:
    x: float
    y: float
    index: int

gt_data = pd.read_csv('data/gt_data.txt', names=['X','Y','Orientation'], sep=' ')
map_data = pd.read_csv('data/map_data.txt', names=['X','Y','# landmark'])
control_data = pd.read_csv('data/control_data.txt', names=['velocity','Yaw rate'], sep=' ')


result = [(x,y, landmark) for x,y,landmark in zip(map_data['X'],map_data['Y'], map_data['# landmark'])]
landarkList=[]
for res in result:
    l = LandMark(res[0],res[1],res[2])
    landarkList.append(l)


a = os.listdir("data/observation")
a.sort()
observation=[]
for i in range(len(a)):
    fileName = 'data/observation/'+a[i]
    observationTmp = pd.read_csv(fileName, names = ['X cord','Y cord'], sep=' ')
    observation.append(observationTmp)

def calculateDistance(landmark1, landmark2):
    a =  np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
    return a


def findClosestLandmark(map_landmarks, singleObs):

    closest_landmark = map_landmarks[0]
    dist = 1000000 # infinity

    for i in range(len(map_landmarks)):
        map_landmark = map_landmarks[i]
        temp_dist = calculateDistance(map_landmark, singleObs)

        if temp_dist < dist:
            dist = temp_dist
            closest_landmark = map_landmark

    return closest_landmark


def getError(gt_data, bestParticle):
    error1 = np.abs(gt_data[0] - bestParticle.x)
    error2 = np.abs(gt_data[1] - bestParticle.y)
    error3 = np.abs(gt_data[2] - bestParticle.theta)
    if(error3>2*np.pi):
        error3 = 2*np.pi - error3
    return (error1, error2, error3)

def findObservationProbability(closest_landmark, map_coordinates, sigmaX, sigmaY):

    mew_x = closest_landmark.x
    mew_y = closest_landmark.y

    x = map_coordinates.x
    y = map_coordinates.y
    frac =  (1/ (2 * np.pi * sigmaX * sigmaY ))
    weight1 = (x-mew_x)**2/((sigmaX)**2)  +(y-mew_y)**2/(sigmaY**2)
    ans = np.exp(-0.5*weight1)
    return ans


def mapObservationToMapCoordinates(observation, particle):
    x = observation.x
    y = observation.y

    xt = particle.x
    yt = particle.y
    theta = particle.theta

    MapX = x * np.cos(theta) - y * np.sin(theta) + xt
    MapY = x * np.sin(theta) + y * np.cos(theta) + yt

    return MapX, MapY

def mapObservationsToMapCordinatesList(observations, particle):

    convertedObservations=[]
    i=0
    for obs in observations.iterrows():
        singleObs = LandMark(obs[1][0],obs[1][1],1)
        mapX, mapY = mapObservationToMapCoordinates(singleObs, particle)
        tmpLandmark = LandMark(x=mapX, y=mapY, index=i)
        i+=1
        convertedObservations.append(tmpLandmark)
    return convertedObservations

class ParticleFilter:
    particles = []
    def __init__(self, intialX, initialY, std, numOfParticles):
        self.number_of_particles = numOfParticles
        for i in range(self.number_of_particles):
            x = random.gauss(intialX, std)
            y = random.gauss(initialY, std)
            theta = random.uniform(0, 2*np.pi)
            tmpParticle = Particle(x,y, theta, 1)
            self.particles.append(tmpParticle)

    def moveParticles(self, velocity, yaw_rate, delta_t=0.1):

        for i in range(self.number_of_particles):
            if (yaw_rate != 0):
                theta = self.particles[i].theta
                newTheta = theta + delta_t * yaw_rate

                newTheta%=2*math.pi

                newX = self.particles[i].x + (velocity / yaw_rate) * (np.sin(newTheta) - np.sin(theta));
                newY = self.particles[i].y + (velocity / yaw_rate) * (np.cos(theta) - np.cos(newTheta));

                self.particles[i].x = newX + random.gauss(0, 0.3)
                self.particles[i].y = newY + random.gauss(0, 0.3)
                self.particles[i].theta = newTheta + random.gauss(0, 0.01)
            else:
                print("ZERO!!!")

    def UpdateWeight(self, observations):
        sigmaX = sigmaY

        map_data_landmarks = []
        # convert map_data to landmarks
        for i in range(len(map_data)):
            map_data_landmarks.append(LandMark(map_data.iloc[i].X, map_data.iloc[i].Y, map_data.iloc[i, 2]))

        # update
        for i in range(self.number_of_particles):
            p_weight = 1.0
            global_observations = mapObservationsToMapCordinatesList(observations, self.particles[i])
            for j in range(len(global_observations)):
                closest_landmark = findClosestLandmark(map_data_landmarks, global_observations[j])
                p_weight *= findObservationProbability(closest_landmark, global_observations[j], sigmaX, sigmaY)
            self.particles[i].weight = p_weight


    def getBestParticle(self):
        best_particle = max(self.particles, key=lambda particle: particle.weight)
        return best_particle

    def getBestParticleOut(self):
        x=0
        y=0
        theta=0
        for i in range(self.number_of_particles):
            x+= self.particles[i].x
            y+= self.particles[i].y
            theta+= self.particles[i].theta
        x=x/self.number_of_particles
        y=y/self.number_of_particles
        theta=theta/self.number_of_particles
        best_particle =  Particle(x,y,theta, weight=1)
        return best_particle

    def PrintWeights(self):
        for i in range(self.number_of_particles):
            print("Weight:",self.particles[i].weight, self.particles[i].x,self.particles[i].y)

    def Resample(self):
        # https://www.youtube.com/watch?v=wNQVo6uOgYA   Resampling Wheel - Artificial Intelligence for Robotics
        # https://www.youtube.com/watch?v=aHLslaWO-AQ

        max_weight = 0
        N = len(self.particles)
        # Get max particle weight
        for i in range(N):
            if (self.particles[i].weight > max_weight):
                max_weight = self.particles[i].weight
                max_index = i

        new_particles = []
        beta = 0.0
        # Guess index uniformaly form all indexes
        index = random.randint(0, N-1)

        # Run resampling
        for i in range(N):
            beta += random.uniform(0, 2 * max_weight)
            while beta > self.particles[index].weight:
                beta -= self.particles[index].weight
                index = (index + 1) % N
            new_particle = Particle(self.particles[index].x, self.particles[index].y, self.particles[index].theta, 1)
            new_particles.append(new_particle)

        self.particles = new_particles



sigmaY = 0.3

magicNumberOfParticles = 200


start = time.time()

def main():
    #particleFilter = ParticleFilter(0 ,0 ,20, numOfParticles=magicNumberOfParticles)
    particleFilter = ParticleFilter(6.2785 ,1.9598 ,0.3, numOfParticles=magicNumberOfParticles)

    for i in range(len(observation)):
        #prediction
        if(i!=0):
            velocity = control_data.iloc[i-1][0]
            yaw_rate = control_data.iloc[i-1][1]
            particleFilter.moveParticles(velocity, yaw_rate)
        a = observation[i].copy()
        particleFilter.UpdateWeight(a)
        particleFilter.Resample()
        bestP = particleFilter.getBestParticle()
        error = getError(gt_data.iloc[i], bestP)
        print(i,error)
    end = time.time()
    print(end - start)

main()
