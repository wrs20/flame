using Random
using Images
using BenchmarkTools
using MPI

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


function save_array_as_image(array, filename)
    N, M = size(array)
    int_array = zeros(UInt8, N, M, 3)
    for colourx in 1:3
        for jx in 1:M
            for ix in 1:N
                fval = 255.0 * array[ix, jx, colourx]
                ival = round(fval)
                ival = max(0, ival)
                ival = min(255, ival)
                int_array[ix, jx, colourx] = ival
            end
        end
    end
    save(filename, int_array)
end


function average_colours(state_colour, rowx, colx, new_colour)
    if (rowx == nothing) || (colx == nothing)
        return
    end
    
    for cx in 1:3
        @inbounds ctmp = 0.5 * state_colour[rowx, colx, cx]
        @inbounds ctmp += 0.5 * new_colour[cx]
        @inbounds state_colour[rowx, colx, cx] = ctmp
    end

end


function reduce_across_ranks(comm, sendbuf)
    recvbuf = MPI.Reduce(sendbuf, MPI.SUM, 0, comm)
    return recvbuf
end


function average_across_ranks(comm, sendbuf)
    recvbuf = reduce_across_ranks(comm, sendbuf)
    SIZE = MPI.Comm_size(comm)
    RANK = MPI.Comm_rank(comm)
    if RANK == 0
        recvbuf /= SIZE
    end
    return recvbuf
end


function point_to_coords(nrow, ncol, x, y)

    x += 1.0
    x *= 0.5

    y += 1.0
    y *= 0.5
    
    x = Int(round(x * ncol))
    if (x<1) || (x>ncol)
        x = nothing
    end

    y = Int(round(y * nrow))
    if (y<1) || (y>ncol)
        y = nothing
    end
    
    return x, y
end


function new_point(rng)
    x = rand(rng)
    y = rand(rng)
    
    x = 2.0 * x - 1.0
    y = 2.0 * y - 1.0

    return x, y
end


function V_0(x, y)
    return x, y
end

function V_1(x, y)
    return sin(x), sin(y)
end

function V_2(x, y)
    r = 1.0 / (x*x + y*y)
    return x * r, y * r
end

struct Variation
    func::Function
    a::Float32
    b::Float32
    c::Float32
    d::Float32
    e::Float32
    f::Float32
end
function(v::Variation)(x, y)
    return v.func(
        v.a * x + v.b * y + v.c, 
        v.d * x + v.e * y + v.f
    )
end

function run_trajectories(N_traj, N_steps, N_burn_in, nrow, ncol, global_state_freq, global_state_colour, funcs, colours, rng)
    for trajx in 1:N_traj
        x, y = new_point(rng)
        
        @inbounds begin
            for stepx in 1:N_steps
                P = rand(rng)
                f = find_ind(P, N_funcs, d_inc_sum)
                x, y = funcs[f](x, y)
                
                if stepx > N_burn_in
                    X, Y = point_to_coords(nrow, ncol, x, y)
                    average_colours(global_state_colour, Y, X, colours[f])
                end
            end
        end
    end
end


MPI.Init()
COMM = MPI.COMM_WORLD
RANK = MPI.Comm_rank(COMM)
SIZE = MPI.Comm_size(COMM)

#funcs = (
#    (V_0,   0.5, [1.0, 0.0, 0.0]),
#    (V_1,   0.5, [0.0, 1.0, 0.0]),
#    (V_2,   0.5, [0.0, 0.0, 1.0]),
#)

# from https://github.com/scottdraves/flam3/blob/master/test.flam3
funcs = (
         (Variation(V_2,-0.681206, -0.0779465, 0.20769 ,  0.755065 , -0.0416126, -0.262334), 0.25, [1.0, 0.0, 0.0]),
         (Variation(V_2, 0.953766,  0.48396  , 0.43268 , -0.0542476,  0.642503 , -0.995898), 0.25, [0.0, 1.0, 0.0]),
         (Variation(V_2, 0.840613, -0.816191 , 0.318971, -0.430402 ,  0.905589 ,  0.909402), 0.25, [0.0, 0.0, 1.0]),
         (Variation(V_2, 0.960492, -0.466555 , 0.215383, -0.727377 , -0.126074 ,  0.253509), 0.25, [0.5, 0.5, 0.0]),
)

if RANK == 0
    println("SETUP...\n")
end

weight_sum = sum([fx[2] for fx in funcs])
d_inc_sum = Array(cumsum([fx[2] / weight_sum for fx in funcs]))

nrow = 2000
ncol = 2000

N_traj = 10000
N_steps = 10000
N_burn_in = 20
N_funcs = length(d_inc_sum)

global_state_freq = Array(zeros(Float32, nrow, ncol));
global_state_colour = Array(zeros(Float32, nrow, ncol, 3));

rng = MersenneTwister(1234);

call_funcs = [fx[1] for fx in funcs]
call_colours = [fx[3] for fx in funcs]

if RANK == 0
    println("RUNNING...\n")
end
run_trajectories(Int(ceil(N_traj/SIZE)), N_steps, N_burn_in, nrow, ncol, global_state_freq, global_state_colour, call_funcs, call_colours, rng)

if RANK == 0
    println("SAVING...")
end

reduce_state_colour = average_across_ranks(COMM, global_state_colour)

if RANK == 0
    save_array_as_image(reduce_state_colour, "foo.png")
end

MPI.Finalize()
