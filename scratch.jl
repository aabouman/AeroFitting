using CSV
using DataFrames
using SparseArrays
using StaticArrays
using LinearAlgebra
using ForwardDiff

# %%
df = CSV.read("processed_data/merged_2021-02-03-16-55-55_seg_2.csv", DataFrame)

p = Matrix(df[!, ["pos x","pos y","pos z"]])
v = Matrix(df[!, ["vel x","vel y","vel z"]])
v̇ = Matrix(df[!, ["acc x","acc y","acc z"]])
ω = Matrix(df[!, ["ang vel x","ang vel y","ang vel z"]])
ω̇ = Matrix(df[!, ["ang acc x","ang acc y","ang acc z"]])
Ω = Matrix(df[!, ["mot 1","mot 2","mot 3","mot 4"]])
Ω̇ = Matrix(df[!, ["dmot 1","dmot 2","dmot 3","dmot 4"]]);

𝕏𝑓 = hcat(v, ω, Ω);
𝕏𝑡 = hcat(v, ω, ω̇, Ω, Ω̇);

size(𝕏𝑡)

# %%
function residual(
    x::MVector{17},
    β::MVector{93},
)
    v, ω, ω̇, Ω, Ω̇ = x[1:3], x[4:6], x[7:9], x[10:13], x[14:17]

    # D ∈ ℝ³ˣ²¹¹
    D = reshape(β[1:87], 3, 29)
    J₁₁, J₁₂, J₁₃, J₂₂, J₂₃, J₃₃ = β[end-5:end]
    𝕁 = [J₁₁  J₁₂  J₁₃;
         J₁₂  J₂₂  J₂₃;
         J₁₃  J₂₃  J₃₃];

    # Body torque
    τ = 𝕁 * ω̇ + ω × (𝕁 * ω)

    # Quadratic model
    y = [v; ω; Ω; Ω̇]
    # z = [1; y; kron(y, y)]
    z = [1; y; y.*y]

    τ̂ = D * z

    r = τ - τ̂
    return r
end

function residual_vec(X::Matrix, β::MVector{93, T}) where {T}
    N = size(X)[1]
    # X = MMatrix{N, 17}(X)
    𝐫 = zeros(T, 3, N)

    println(N)
    for i in 1:N
        # 𝐫[:, i] .= residual(X[i,:], β)
        𝐫[:, i] .= residual(MVector{17}(X[i,:]), β)
    end

    return vec(𝐫)
end

function cost(X, β)
    𝐫 = residual_vec(X, β)
    S = 𝐫' * 𝐫
    return S
end

# %%
β = @MVector rand(93)
test = 𝕏𝑡[1:1000:end,:]
𝐫 = residual_vec(test, β)
display(𝐫)

# %%
Jᵣ = ForwardDiff.jacobian(_β->residual_vec(test, _β), β)
β += (Jᵣ \ residual_vec(test, β))

# %%
J₁₁, J₁₂, J₁₃, J₂₂, J₂₃, J₃₃ = (Jᵣ \ residual_vec(test, β))[end-5:end]
𝕁 = [J₁₁  J₁₂  J₁₃;
     J₁₂  J₂₂  J₂₃;
     J₁₃  J₂₃  J₃₃];
display(𝕁)
