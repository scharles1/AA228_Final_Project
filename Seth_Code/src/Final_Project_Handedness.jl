using LightGraphs
using DataFrames
using DataStructures
using Iterators
using StatsBase
using Distributions

############################################################
# Function: write_Policy(dag::DiGraph, idx2names, filename)
#
# Description: 
############################################################
function write_Policy(policy::Array{Int64,1}, score::Float64)
    open(outputfilename, "a") do io
        @printf(io, "Policy Score: %f\n", score)
        @printf(io, "Baseline Score: %f\n", baseline)
        for s= 1:kNumStates
            @printf(io, "%d\n", policy[s])
        end
    end
end

############################################################
# Function: computePolicy
#
# Description:
############################################################
function computePolicy()   
    Pi, score = QLearning()
    #Pi, score = Sarsa()
    #Pi, score = BayesQLearning()
    write_Policy(Pi, score)
end

############################################################
# Function: computeBaselineScore
#
# Description:
############################################################
function computeBaselineScore()
    aveScore = 0.0
    n = 100
    for i = 1:n
        Pi = rand(1:(kNumActions-1),kNumStates)
        score = computePolicyScore(Pi)
        @printf("Random Policy Score: %f\r\n", score)
        aveScore = aveScore + score
    end
    aveScore = aveScore/n
    @printf("Average of 100 Random Policy Scores: %f\r\n", aveScore)  
    return aveScore
end

############################################################
# Function: computePolicyScore
#
# Description:
############################################################
function computePolicyScore(Pi::Array{Int64,1})
    U = zeros(kNumStates,1)
    for i = 1:100
        #U_old = copy(U)
        for s = 1:kNumStates
            U[s] = R[s,Pi[s]] + sum(T[:,s,Pi[s]].*U[:])
        end
        #difference = U - U_old
        #@printf("Iteration: %i\tL2-norm of difference: %f\r\n", i, dot(difference,difference))
    end
    return sum(U)
end

############################################################
# Function: QLearning
#
# Description:
############################################################
function QLearning()
    Q = zeros(kNumStates, kNumActions)
    Q_old = ones(kNumStates, kNumActions)    
    Pi = Array{Int64,1}(kNumStates)
    s = 1
    converged = 0
    i = 0

    #for k = 1:kIterations
    while(converged == 0)
        if ((i%10) == 0)
            a = rand(1:kNumActions)
        else
            a = indmax(Q[s,:])
        end
        sp = sample(collect(1:kNumStates), ProbabilityWeights(T[:,s,a]))
        r = computeRunsScored(s,sp)
        Q[s,a] = Q[s,a] + kLearnFactor*(r + kDiscountFactor*maximum(Q[sp,:]) - Q[s,a])
        
        if ((i % 100000) == 0)
            difference = norm(Q - Q_old,2)
            @printf("Iteration: %i\tL2-norm of difference: %f\r\n", i, difference)
            Q_old = copy(Q)
            if (difference < 2.5)
                converged = 1
            end
        end

        if(sp == 289)   
            s = 1
        else
            s = sp
        end
        i = i + 1
    end

    for s = 1:kNumStates
        Pi[s] = indmax(Q[s,:])
    end

    score = computePolicyScore(Pi)
    return Pi, score
end

############################################################
# Function: Sarsa
#
# Description:
############################################################
function Sarsa()
    Q = zeros(kNumStates, kNumActions)
    Pi = Array{Int64,1}(kNumStates)
    s = 1
    a = rand(1:kNumActions)

    for k = 1:kIterations
        ap = rand(1:kNumActions)
        sp = sample(collect(1:kNumStates), ProbabilityWeights(T[:,s,a]))
        r = R[s,a]
        Q[s,a] = Q[s,a] + kLearnFactor*(r + kDiscountFactor*Q[sp,ap] - Q[s,a])
        a = ap  
        if(sp == 289)   
            s = 0
        else
            s = sp
        end
    end

    for s = 1:kNumStates
        Pi[s] = indmax(Q[s,:])
    end

    score = computePolicyScore(Pi)
    return Pi, score
end

############################################################
# Function: BayesQLearning
#
# Description:
############################################################
function BayesQLearning()
    Q = zeros(kNumStates, kNumActions)
    Tdir = Array{Dirichlet{Float64},2}(kNumStates,kNumActions)
    for s = 1:kNumStates
    	for a = 1:kNumActions
            prior = ones(kNumStates)*0.00001 
            if(maximum(N[s,a,:]) == 0)
                prior = ones(kNumStates)
            end
    		Tdir[s,a] = Dirichlet((prior + N[s,a,:]))
     	end
    end

    Pi = Array{Int64,1}(kNumStates)
    states = collect(1:kNumStates)
    numGamesPlayed = 0
    s = 1

    while (numGamesPlayed < kNumGames2Play)
        a = rand(1:kNumActions)
        sp = sample(states, ProbabilityWeights(rand(Tdir[s,a])))
        r =  computeRunsScored(s,sp)
        Q[s,a] = Q[s,a] + kLearnFactor*(r + kDiscountFactor*maximum(Q[sp,:]) - Q[s,a])
        if(sp == 289)   
            numGamesPlayed = numGamesPlayed + 1
            s = 1
        else
            s = sp
        end
    end

    for s = 1:kNumStates
        Pi[s] = indmax(Q[s,:])
    end

    score = computePolicyScore(Pi)
    return Pi, score
end

function computeRunsScored(s,sp)
    sNumRunners = numRunners((s-1) % 8)
    spNumRunners = numRunners((sp-1) % 8)
    sNumOuts = div((s-1),96)
    spNumOuts = div((sp-1),96)
    return sNumRunners - spNumRunners + 1 - spNumOuts + sNumOuts
end

function numRunners(runnerCode)
    if (runnerCode == 0)
        return 0
    elseif (runnerCode in (1, 2, 3))
        return 1
    elseif (runnerCode in (4, 5, 6))
        return 2
    else
        return 3
    end
end
############################################################
# Function: computeCountsMatrix
#
# Description:
############################################################
function computeCountsMatrix(data::Array{Int64,2})
    N = zeros(Int64,kNumStates, kNumActions, kNumStates)
    for i = 1:length(data[:,1])
        s = data[i,1]
        a = data[i,2]
        sp = data[i,4]
        N[s,a,sp] = N[s,a,sp] + 1
    end
    return N
end

############################################################
# Function: computeTransitionMatrix
#
# Description:
############################################################
function computeTransitionMatrix()
    T = zeros(kNumStates, kNumStates, kNumActions)
    for sp = 1:kNumStates
    @printf("sp: %i\r\n", sp)
    for s = 1:kNumStates, a = 1:kNumActions
        total = sum(N[s,a,:])
        if(total == 0)
            total = 1
        end
        T[sp,s,a] = N[s,a,sp]/total
    end
    end
    T[:,:,64] = eye(kNumStates, kNumStates)
    return T
end

############################################################
# Function: computeRewardMatrix
#
# Description:
############################################################
function computeRewardMatrix(data::Array{Int64,2})
    R = zeros(kNumStates, kNumActions)
    counts = zeros(kNumStates, kNumActions)
    for i = 1:length(data[:,1])
        s = data[i,1]
        a = data[i,2]
        r = data[i,3]
        counts[s,a] = counts[s,a] + 1
        rhat = r/sum(N[s,a,:])
        R[s,a] = R[s,a] + (rhat - R[s,a])/counts[s,a]
    end
    R[:,64] = ones(kNumStates)*switch_Penalty
    return R
end

#definitions
inputfilename = "../obs3.csv"
outputfilename = "./Results_Handedness/Q_learning_Results/Checking_Convergence1.policy"
kNumActions = 64
kNumStates = 1153
switch_Penalty = -1.0
kDiscountFactor = 0.9
const kLearnFactor = 0.1
const kIterations = 1000000000
const kNumGames2Play = 10000000
baseline = -129.140887

#data = readtable(inputfilename)
#data = convert(Array{Int64}, data)
#N = computeCountsMatrix(data)
#T = computeTransitionMatrix()
#R = computeRewardMatrix(data)
#baseline = computeBaselineScore()
computePolicy()