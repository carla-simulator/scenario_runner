#!/usr/bin/env groovy

pipeline
{

    agent any
    
    options
    {
        buildDiscarder(logRotator(numToKeepStr: '3', artifactNumToKeepStr: '3'))
    }

    stages
    {
        stage('Build Package')
        {
            steps
            {
                sh 'echo "Building scenario runner package"' 
            }
        }
        stage('Deploy CARLA')
        {
            steps
            {
                sh 'echo "Deploying CARLA server"' 
            }
        }
        stage('Make Tests')
        {
            steps
            {
                sh 'echo "Making tests"' 
            }
        }
    }
}

