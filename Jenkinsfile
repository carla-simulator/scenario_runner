#!/usr/bin/env groovy


String CARLA_HOST 
String CARLA_RELEASE
String TEST_HOST
String COMMIT
String ECR_REPOSITORY = "456841689987.dkr.ecr.eu-west-3.amazonaws.com/scenario_runner"
boolean CARLA_RUNNING = false
boolean CONCURRENCY = true

// V3 - include detection of concurrent builds

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
                        script: "sed -e 's/^\s*HOST\s*=\s*//;t;d' ./CARLA_VER",
                        returnStdout: true).trim()
                    CARLA_RELEASE = sh(
                        script: "sed -e 's/^\s*RELEASE\s*=\s*//;t;d' ./CARLA_VER",
                        returnStdout: true).trim()
                    COMMIT = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                }
                println "using ${CARLA_RELEASE} from ${TEST_HOST}"
            }
        }
        stage('enter gpu slot')
        {
            options
            {
                lock resource: 'ubuntu_gpu_slot', inversePrecedence: true
            }
            stages
            {
                stage('lider build')
                {
                    options
                    {
                        lock resource: 'ubuntu_gpu', skipIfLocked: true
                    }
                    stages
                    {
                        stage('start gpu slave')
                        {
                            agent { label "master" }
                            steps
                            {
                                script
                                {
                                    println "No concurrency or leader buid. Starting gpu slave."
                                    CONCURRENCY = false
                                    aux_lib = load("/home/jenkins/aux.groovy")
                                    jenkins_lib = load("/home/jenkins/scenario_runner.groovy")
                                    println "open queue entering."
                                    aux_lib.unlock("ubuntu_gpu_slot")
                                    jenkins_lib.StartUbuntuTestNode()
                                }
                            }
                        }
                        stage('deploy')
                        {
                            parallel
                            {
                                stage('build image')
                                {
                                    options
                                    {
                                        lock resource: "docker_build"
                                    }
                                    agent { label "master" }
                                    steps
                                    {
                                        script
                                        {
                                            sh "docker build -t jenkins/scenario_runner:${COMMIT} ."
                                            sh "docker tag jenkins/scenario_runner:${COMMIT} ${ECR_REPOSITORY}:${COMMIT}"
                                            sh '$(aws ecr get-login | sed \'s/ -e none//g\' )'
                                            sh "docker push ${ECR_REPOSITORY}:${COMMIT}"
                                            sh "docker image rmi -f \"\$(docker images -q ${ECR_REPOSITORY}:${COMMIT})\""
                                            sh 'docker system prune --volumes -f'
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
                        stage('test')
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
                }
                stage('get gpu slot')
                {
                    when
                    {
                        expression 
                        { 
                            return CONCURRENCY
                        }
                    }
                    options
                    {
                        lock resource: 'ubuntu_gpu', inversePrecedence: true
                    }
                    stages
                    {
                        stage('deploy')
                        {
                            parallel
                            {
                                stage('build SR docker image')
                                {
                                    options
                                    {
                                        lock resource: "docker_build"
                                    }
                                    agent { label "master" }
                                    steps
                                    {
                                        script
                                        {
                                            sh "docker build -t jenkins/scenario_runner:${COMMIT} ."
                                            sh "docker tag jenkins/scenario_runner:${COMMIT} ${ECR_REPOSITORY}:${COMMIT}"
                                            sh '$(aws ecr get-login | sed \'s/ -e none//g\' )'
                                            sh "docker push ${ECR_REPOSITORY}:${COMMIT}"
                                            sh "docker image rmi -f \"\$(docker images -q ${ECR_REPOSITORY}:${COMMIT})\""
                                            sh 'docker system prune --volumes -f'
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
                        stage('test')
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
                                    echo 'end'
                                    deleteDir()
                                }
                            }
                        }
                    }
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
                        if ( CONCURRENCY == false )
                        {
                            lock('ubuntu_gpu_slot')
                            {
                                jenkins_lib = load("/home/jenkins/scenario_runner.groovy")
                                lock('ubuntu_gpu')
                                {
                                    jenkins_lib.StopUbuntuTestNode()
                                    echo 'leader build stopped'
                                }
                            }
                        }
                    }
                    deleteDir()
                }
        }
    }
}
