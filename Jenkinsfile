#!/usr/bin/env groovy

/**
	Version 2 
	---------

	This version deploys scenario runner from the just
	stored image in a ECR repository

**/

String CARLA_HOST 
String CARLA_RELEASE
String TEST_HOST
String COMMIT
String ECR_REPOSITORY = "456841689987.dkr.ecr.eu-west-3.amazonaws.com/scenario_runner"
boolean CARLA_RUNNING = false

pipeline
{
    agent none

    options
    {
        buildDiscarder(logRotator(numToKeepStr: '3', artifactNumToKeepStr: '3'))
        skipDefaultCheckout()
    }

    stages
    {
        stage('setup')
        {
            agent { label "master" }
            steps
            {
                checkout scm
                script
                {
                    jenkinsLib = load("/home/jenkins/scenario_runner.groovy")
                    TEST_HOST = jenkinsLib.getUbuntuTestNodeHost()
                    CARLA_HOST= sh(
                        script: "cat ./CARLA_VER | grep HOST | sed 's/HOST\\s*=\\s*//g'",
                        returnStdout: true).trim()
                    CARLA_RELEASE = sh(
                        script: "cat ./CARLA_VER | grep RELEASE | sed 's/RELEASE\\s*=\\s*//g'",
                        returnStdout: true).trim()
		    COMMIT = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                }
                println "using CARLA version ${CARLA_RELEASE} from ${TEST_HOST}"
            }
        }
        stage('deploy')
        {
            parallel
            {
                stage('build SR docker image')
                {
                    agent { label "master" }
                    steps
                    {
                        //checkout scm
                        sh 'docker build -t jenkins/scenario_runner .'
                        sh "docker tag jenkins/scenario_runner ${ECR_REPOSITORY}:${COMMIT}"
                        sh '$(aws ecr get-login | sed \'s/ -e none//g\' )' 
                        sh "docker push ${ECR_REPOSITORY}"
                    }
                }
                stage('deploy CARLA')
                {
                    stages
                    {
                        stage('start server')
                        {
                            agent { label "master" }
                            steps
                            {
                                script
                                {
                                    jenkinsLib = load("/home/jenkins/scenario_runner.groovy")
                                    jenkinsLib.StartUbuntuTestNode()
                                }
                            }
                        }
                        stage('install CARLA')
                        {
                            agent { label "slave && ubuntu && gpu && sr" }
                            steps
                            {
                                println "using CARLA version ${CARLA_RELEASE}"
                                sh "wget -qO- ${CARLA_HOST}/${CARLA_RELEASE}.tar.gz | tar -xzv -C ."
                            }
                        }
                    }
                }
            }
        }
        stage('run test')
        {
        	agent { label "slave && ubuntu && gpu && sr" }
            steps
            {
                sh 'DISPLAY= ./CarlaUE4.sh -opengl -nosound > CarlaUE4.log&'
        sleep 10
                script
                {
                        sh '$(aws ecr get-login | sed \'s/ -e none//g\' )' 
                        sh "docker pull ${ECR_REPOSITORY}:${COMMIT}"
                        sh "docker container run --rm --network host -e LANG=C.UTF-8 \"${ECR_REPOSITORY}:${COMMIT}\" -c \"python3 scenario_runner.py --scenario FollowLeadingVehicle_1 --debug --output --reloadWorld \""
                        deleteDir()
                }
            }
        }
    }
    post
    {
        always
        {
            node('master')
            {
                script  
                {
                    jenkinsLib = load("/home/jenkins/scenario_runner.groovy")
                    jenkinsLib.StopUbuntuTestNode()
                    echo 'test node stopped'
                    sh 'docker system prune --volumes -f'
                }
                deleteDir()
            }
        }
    }
}
