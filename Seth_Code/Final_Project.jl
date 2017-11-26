using LightGraphs
using DataFrames
using DataStructures
using Iterators
using StatsBase

############################################################
# Function: write_Policy(dag::DiGraph, idx2names, filename)
#
# Description: 
############################################################
function write_Policy(policy::Array{Int64,1}, score::Float64)
    open(outputfilename, "w") do io
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
    write_Policy(Pi, score)
end

############################################################
# Function: computeBaselineScore
#
# Description:
############################################################
function computeBaselineScore()
    aveScore = 0.0
    for i = 1:100
        Pi = rand(1:kNumActions,kNumStates)
        score = computePolicyScore(Pi)
        @printf("Random Policy Score: %f\r\n", score)
        aveScore = aveScore + (score - aveScore)/i
    end
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
        for s = 1:kNumStates
            U[s] = R[s,Pi[s]] + sum(T[:,s,Pi[s]].*U[:])
        end
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
    Pi = Array{Int64,1}(kNumStates)
    s = rand(1:kNumStates)

    for k = 1:kIterations
        a = rand(1:kNumActions)
        sp = sample(collect(1:kNumStates), ProbabilityWeights(T[:,s,a]))
        r = R[s,a]
        Q[s,a] = Q[s,a] + kLearnFactor*(r + kDiscountFactor*maximum(Q[sp,:]) - Q[s,a])
        s = sp
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
    s = rand(1:kNumStates)
    a = rand(1:kNumActions)

    for k = 1:kIterations
        ap = rand(1:kNumActions)
        sp = sample(collect(1:kNumStates), ProbabilityWeights(T[:,s,a]))
        r = R[s,a]
        Q[s,a] = Q[s,a] + kLearnFactor*(r + kDiscountFactor*Q[sp,ap] - Q[s,a])
        a = ap        
        s = sp
    end

    for s = 1:kNumStates
        Pi[s] = indmax(Q[s,:])
    end

    score = computePolicyScore(Pi)
    return Pi, score
end

############################################################
# Function: computeCountsMatrix
#
# Description:
############################################################
function computeCountsMatrix(data::DataFrame)
    N = zeros(kNumStates, kNumActions, kNumStates)
    for i = 1:length(data[1])
        s = data[i,1] + 1
        a = data[i,2] + 1
        sp = data[i,4] + 1
        N[s, a, sp] = N[s, a, sp] + 1
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
        for s = 1:kNumStates
            for a = 1:kNumActions
                total = sum(N[s,a,:])
                if(total == 0)
                    total = 1
                end
                T[sp,s,a] = N[s,a,sp]/total
            end
        end
    end
    return T
end

############################################################
# Function: computeRewardMatrix
#
# Description:
############################################################
function computeRewardMatrix(data::DataFrame)
    R = zeros(kNumStates, kNumActions)
    counts = zeros(kNumStates, kNumActions)
    for i = 1:length(data[1])
        s = data[i,1] + 1
        a = data[i,2] + 1
        r = data[i,3]
        counts[s,a] = counts[s,a] + 1
        rhat = r/sum(N[s,a,:])
        R[s,a] = R[s,a] + (rhat - R[s,a])/counts[s,a]
    end
    return R
end

############################################################
# Function: initMatrices
#
# Description:
############################################################
function initMatrices(data::DataFrame)
    return computeCountsMatrix(data), computeTransitionMatrix(), computeRewardMatrix(data)
end

#definitions
inputfilename = "../obs.csv"
outputfilename = "../final_project.policy"
const kNumActions = 63
const kNumStates = 288
const kDiscountFactor = 1
const kLearnFactor = 0.5
const kIterations = 100000000

#data = readtable(inputfilename)
#N,T,R = initMatrices(data)
#baseline = computeBaselineScore()
computePolicy()