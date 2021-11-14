using Random
using Images


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
    state_colour[rowx, colx, :] = 0.5 * (state_colour[rowx, colx, :] + new_colour)
end

function average_thread_colours(global_state_colour, state_colour)
    N, M, _, N_threads = size(global_state_colour)
    for threadx in 1:N_threads
        state_colour[:, :, :] += global_state_colour[:, :, :, threadx]
    end
    state_colour[:] /= N_threads
end


function point_to_coords(nrow, ncol, x, y)

    x += 1.0
    x *= 0.5

    y += 1.0
    y *= 0.5
    
    x = Int(round(x * ncol))
    x = max(1, x)
    x = min(ncol, x)
    y = Int(round(y * nrow))
    y = max(1, y)
    y = min(nrow, y)
    
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
    func
    a
    b
    c
    d
    e
    f
end
function(v::Variation)(x, y)
    return v.func(
        v.a * x + v.b * y + v.c, 
        v.d * x + v.e * y + v.f
    )
end


funcs = (
    (V_0,   0.5, [1.0, 0.0, 0.0]),
    (V_1,   0.5, [0.0, 1.0, 0.0]),
    (V_2, 0.5, [0.0, 0.0, 1.0]),
)


funcs = (
         (Variation(V_2,-0.681206, -0.0779465, 0.20769 ,  0.755065 , -0.0416126, -0.262334), 0.25, [1.0, 0.0, 0.0]),
         (Variation(V_2, 0.953766,  0.48396  , 0.43268 , -0.0542476,  0.642503 , -0.995898), 0.25, [0.0, 1.0, 0.0]),
         (Variation(V_2, 0.840613, -0.816191 , 0.318971, -0.430402 ,  0.905589 ,  0.909402), 0.25, [0.0, 0.0, 1.0]),
         (Variation(V_2, 0.960492, -0.466555 , 0.215383, -0.727377 , -0.126074 ,  0.253509), 0.25, [0.5, 0.5, 0.0]),
)


weight_sum = sum([fx[2] for fx in funcs])
d_inc_sum = Array(cumsum([fx[2] / weight_sum for fx in funcs]))

@show d_inc_sum

nrow = 1000
ncol = 1000

N_traj = 2000
N_steps = 10000
N_burn_in = 100
N_funcs = length(d_inc_sum)


N_threads = Threads.nthreads()
global_state_freq = Array(zeros(Float32, nrow, ncol, N_threads));
global_state_colour = Array(zeros(Float32, nrow, ncol, 3, N_threads));

state_freq = Array(zeros(Float32, nrow, ncol));
state_colour = Array(zeros(Float32, nrow, ncol, 3));

rng = MersenneTwister(1234);

Threads.@threads for trajx in 1:N_traj
    
    thread_id = Threads.threadid()
    state_freq = view(global_state_freq, :, :, thread_id)
    state_colour = view(global_state_colour, :, :, :, thread_id)

    x, y = new_point(rng)

    for stepx in 1:N_steps
        P = rand(rng)
        f = find_ind(P, N_funcs, d_inc_sum)
        x, y = funcs[f][1](x, y)
        if stepx > N_burn_in
            X, Y = point_to_coords(nrow, ncol, x, y)
            average_colours(state_colour, Y, X, funcs[f][3])
        end
    end
end


average_thread_colours(global_state_colour, state_colour)
save_array_as_image(state_colour, "foo.png")


