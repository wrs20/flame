using CUDA
using Random
using GPUArrays

function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end


function find_ind(value, N_funcs, d_inc_sum)
    @inbounds begin
        for ind = 1:N_funcs
            if value <= d_inc_sum[ind]
                return ind
            end
        end
        return N_funcs
    end
end




# determine the function call order from the random numbers
function gpu_bin_func(N_traj, N_steps, rand_numbers, N_funcs, d_inc_sum, d_func_order)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    iy = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if (ix <= N_traj)
        for sx in 1:N_steps
            @inbounds begin
                ind = find_ind(rand_numbers[ix, sx], N_funcs, d_inc_sum)
                d_func_order[ix, sx] = ind
            end
        end
    end

    return nothing
end






function one()
    return 1
end

function two()
    return 2
end



funcs = (
    0.5 => one,
    0.5 => two,
)


weight_sum = sum([fx.first for fx in funcs])
d_inc_sum = CuArray(cumsum([fx.first / weight_sum for fx in funcs]))
h_func_handles = Array([fx.second for fx in funcs])

@show d_inc_sum
@show h_func_handles



N = 3
M = 4

N_traj = 10
N_steps = 10
N_funcs = length(d_inc_sum)

state_freq = CuArray(zeros(Float32, N, M));
state_colour = CuArray(zeros(Float32, N, M, 3));

rand_numbers = CuArray(zeros(Float32, N_traj, N_steps));
d_func_order = CuArray(zeros(Int32, N_traj, N_steps));
Random.rand!(GPUArrays.default_rng(CuArray), rand_numbers);


numblocks = ceil(Int, N_traj/256)

@cuda threads=256 blocks=numblocks gpu_bin_func(N_traj, N_steps, rand_numbers, N_funcs, d_inc_sum, d_func_order)
synchronize()

@show d_func_order


traj = CuArray(zeros(Float32, N_traj, 2))



