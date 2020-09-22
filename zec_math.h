/*
    Inspired by Sean Barrett's `stb` libraries:
    https://github.com/nothings/stb

    Do this:
        #define ZEC_MATH_IMPLEMENTATION
    before you include this file in *one* C or C++ file to create the implementation.
       
    // i.e. it should look like this:
    #include ...
    #define ZEC_MATH_IMPLEMENTATION
    #include "zec_math.h"
    
    # Notes:

    - Vectors are _column-major_ and we use _post-multiplication_
    so to transform a vector you would do M * v
    - Matrices are layed out in row-major form, so data[1][2] refers to the third element
    in the second row (A_2,3)
    - To use the matrices produced by this lib with HLSL youll have to enable row-major storage.
*/

#ifndef ZEC_INCLUDE_MATH_H
#define ZEC_INCLUDE_MATH_H
#include <cmath>
#include <cstring>

namespace zec
{
    constexpr float k_pi = M_PI;
    constexpr float k_half_pi = M_PI_2;
    constexpr float k_quarter_pi = M_PI_4;
    constexpr float k_inv_pi = M_1_PI;
    constexpr float k_2_pi = M_PI * 2.0f;

    inline float deg_to_rad(const float degrees)
    {
        constexpr float conversion_factor = k_pi / 180.0f;
        return degrees * conversion_factor;
    }

    inline float rad_to_deg(const float radians)
    {
        constexpr float conversion_factor = 180.0f * k_inv_pi;
        return radians * conversion_factor;
    }

    // ---------- vec3 ----------

    struct vec3
    {
        union
        {
            float data[3] = {};
            struct
            {
                float x, y, z;
            };
            struct
            {
                float r, g, b;
            };
        };

        inline float &operator[](const size_t idx)
        {
            return data[idx];
        }

        inline const float &operator[](const size_t idx) const
        {
            return data[idx];
        }
    };

    constexpr vec3 k_up = {0.0f, 1.0f, 0.0f};

    inline vec3 operator+(const vec3 &v1, const vec3 &v2);
    inline vec3 operator-(const vec3 &v1);
    inline vec3 operator-(const vec3 &v1, const vec3 &v2);
    inline vec3 operator*(const vec3 &v1, const float s);
    inline vec3 operator*(const float s, const vec3 &v1);
    inline vec3 operator/(const vec3 &v1, const float s);

    inline vec3 &operator+=(vec3 &v1, const vec3 &v2);
    inline vec3 &operator*=(vec3 &v1, const float s);

    inline float length_squared(vec3 v);
    inline float length(vec3 v);
    inline vec3 normalize(vec3 v);

    float dot(const vec3 &a, const vec3 &b);
    vec3 cross(const vec3 &a, const vec3 &b);

    // ---------- vec4 ----------

    struct vec4
    {
        union
        {
            float data[4] = {};
            struct
            {
                float x, y, z, w;
            };
            struct
            {
                float r, g, b, a;
            };
            struct
            {
                vec3 xyz;
                float w;
            };
            struct
            {
                vec3 rgb;
                float a;
            };
        };
        vec4() : data{0.0f, 0.0f, 0.0f, 0.0f} {};
        vec4(const float x, const float y, const float z, const float w) : x(x), y(y), z(z), w(w){};
        vec4(vec3 xyz, const float w = 1.0f) : xyz(xyz), w(w){};

        inline float &operator[](const size_t idx)
        {
            return data[idx];
        }

        inline const float &operator[](const size_t idx) const
        {
            return data[idx];
        }
    };

    inline vec4 operator+(const vec4 &v1, const vec4 &v2);
    inline vec4 operator*(const vec4 &v1, const float s);
    inline vec4 operator/(const vec4 &v1, const float s);

    inline float length_squared(vec4 v);
    inline float length(vec4 v);
    inline vec4 normalize(vec4 v);

    //

    struct mat3
    {
        union
        {
            vec3 rows[3] = {};
            float data[3][3];
        };

        mat3() : rows{{}, {}, {}} {};
        mat3(const vec3 &r1, const vec3 &r2, const vec3 &r3)
        {
            memcpy(&rows[0], &r1, sizeof(r1));
            memcpy(&rows[1], &r2, sizeof(r1));
            memcpy(&rows[2], &r3, sizeof(r1));
        }
        mat3(const vec3 in_rows[3])
        {
            memcpy(data, in_rows, sizeof(data));
        }

        inline vec3 column(const size_t col_idx) const
        {
            return {rows[0][col_idx], rows[1][col_idx], rows[2][col_idx]};
        }

        inline float *operator[](const size_t index)
        {
            return data[index];
        }

        inline const float *operator[](const size_t index) const
        {
            return data[index];
        }
    };

    mat3 transpose(const mat3 &m);

    vec3 operator*(const mat3 &m, const vec3 &v);

    // ---------- mat4 ----------

    struct mat4
    {
        union
        {
            vec4 rows[4] = {};
            float data[4][4];
        };

        mat4() : rows{{}, {}, {}, {}} {};
        mat4(const vec4 &r1, const vec4 &r2, const vec4 &r3, const vec4 &r4)
        {
            memcpy(&rows[0], &r1, sizeof(r1));
            memcpy(&rows[1], &r2, sizeof(r1));
            memcpy(&rows[2], &r3, sizeof(r1));
            memcpy(&rows[3], &r4, sizeof(r1));
        }
        mat4(const vec4 in_rows[4])
        {
            memcpy(data, in_rows, sizeof(data));
        }
        mat4(const mat3 &rotation, const vec3 &translation)
        {
            memcpy(&rows[0], rotation[0], sizeof(vec3));
            memcpy(&rows[1], rotation[1], sizeof(vec3));
            memcpy(&rows[2], rotation[2], sizeof(vec3));
            rows[3] = {0.0f, 0.0f, 0.0f, 1.0f};
            data[0][3] = translation.x;
            data[1][3] = translation.y;
            data[2][3] = translation.z;
        }

        inline vec4 column(const size_t col_idx) const
        {
            return {rows[0][col_idx], rows[1][col_idx], rows[2][col_idx], rows[3][col_idx]};
        }

        inline float *operator[](const size_t index)
        {
            return data[index];
        }

        inline const float *operator[](const size_t index) const
        {
            return data[index];
        }
    };

    mat4 operator*(const mat4 &m1, const mat4 &m2);
    mat4 &operator/=(mat4 &m, const float s);

    vec4 operator*(const mat4 &m, const vec4 &v);

    inline mat4 identity_mat4();

    vec3 get_right(const mat4 &m);
    vec3 get_up(const mat4 &m);
    vec3 get_dir(const mat4 &m);

    mat4 look_at(vec3 pos, vec3 origin, vec3 up);

    mat4 invert(const mat4 &m);

    mat4 transpose(const mat4 &m);

    mat3 to_mat3(const mat4 &m);

    // ---------- quaternion ----------

    struct quaternion
    {
        union
        {
            float data[4] = {};
            struct
            {
                float x, y, z, w;
            };
            struct
            {
                vec3 v;
                float w;
            };
        };

        quaternion() = default;
        quaternion(const float x, const float y, const float z, const float w) : x(x), y(y), z(z), w(w){};
        quaternion(vec3 v, float w) : v(v), w(w){};

        inline float &operator[](const size_t idx)
        {
            return data[idx];
        }

        inline const float &operator[](const size_t idx) const
        {
            return data[idx];
        }
    };

    inline quaternion operator*(const quaternion &q1, const float s);
    inline quaternion operator/(const quaternion &q1, const float s);

    inline quaternion from_axis_angle(const vec3 &axis, const float angle);

    inline float length_squared(quaternion v);
    inline float length(quaternion v);
    inline quaternion normalize(quaternion v);

    mat4 quat_to_mat(const quaternion &q);

    // ---------- Transformation helpers ----------

    inline void rotate(mat4 &mat, const quaternion &q);

    inline void set_translation(mat4 &mat, const vec3 &translation);
    //mat4 orthogonal_projection(float left, float right, float near, float far, float top, float bottom)

    // Aspect ratio is in radians, please
    mat4 perspective_projection(const float aspect_ratio, const float fov, const float z_near, const float z_far);
} // namespace zec
#endif //  ZEC_INCLUDE_MATH_H

#ifdef ZEC_MATH_IMPLEMENTATION
namespace zec
{
    inline vec3 operator+(const vec3 &v1, const vec3 &v2)
    {
        return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
    }

    inline vec3 operator-(const vec3 &v1)
    {
        return {-v1.x, -v1.y, -v1.z};
    }

    inline vec3 operator-(const vec3 &v1, const vec3 &v2)
    {
        return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    }

    inline vec3 operator*(const vec3 &v1, const float s)
    {
        return {v1.x * s, v1.y * s, v1.z * s};
    }

    inline vec3 operator*(const float s, const vec3 &v1)
    {
        return {v1.x * s, v1.y * s, v1.z * s};
    }

    inline vec3 operator/(const vec3 &v1, const float s);
    {
        float inv_s = 1.0f / s;
        return {v1.x * inv_s, v1.y * inv_s, v1.z * inv_s};
    }

    inline vec3 &operator+=(vec3 &v1, const vec3 &v2)
    {
        v1.x += v2.x;
        v1.y += v2.y;
        v1.z += v2.z;
        return v1;
    }

    inline vec3 &operator*=(vec3 &v1, const float s)
    {
        v1.x *= s;
        v1.y *= s;
        v1.z *= s;
        return v1;
    }

    inline float length_squared(vec3 v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    inline float length(vec3 v)
    {
        return sqrtf(length_squared(v));
    }

    inline vec3 normalize(vec3 v)
    {
        return v / length(v);
    }

    inline vec4 operator+(const vec4 &v1, const vec4 &v2)
    {
        return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w};
    }

    inline vec4 operator*(const vec4 &v1, const float s)
    {
        return {v1.x * s, v1.y * s, v1.z * s, v1.w * s};
    }

    inline vec4 operator/(const vec4 &v1, const float s)
    {
        float inv_s = 1.0f / s;
        return {v1.x * inv_s, v1.y * inv_s, v1.z * inv_s, v1.w * inv_s};
    }

    inline float length_squared(vec4 v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    inline float length(vec4 v)
    {
        return sqrtf(length_squared(v));
    }

    inline vec4 normalize(vec4 v)
    {
        return v / length(v);
    }

    mat4 quat_to_mat(const quaternion &q)
    {
        quaternion nq = normalize(q);
        float qx2 = nq.x * nq.x;
        float qy2 = nq.y * nq.y;
        float qz2 = nq.z * nq.z;
        float qxy = nq.x * nq.y;
        float qxz = nq.x * nq.z;
        float qxw = nq.x * nq.w;
        float qyz = nq.y * nq.z;
        float qyw = nq.y * nq.w;
        float qzw = nq.z * nq.w;

        return {
            {1 - 2 * (qy2 + qz2), 2 * (qxy - qzw), 2 * (qxz + qyw), 0},
            {2 * (qxy + qzw), 1 - 2 * (qx2, qz2), 2 * (qyz - qxw), 0},
            {2 * (qxz - qyw), 2 * (qyz + qxw), 1 - 2 * (qx2 + qy2), 0},
            {0, 0, 0, 1},
        };
    }

    float dot(const vec3 &a, const vec3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    vec3 cross(const vec3 &a, const vec3 &b)
    {
        float a1b2 = a[0] * b[1];
        float a1b3 = a[0] * b[2];
        float a2b1 = a[1] * b[0];
        float a2b3 = a[1] * b[2];
        float a3b1 = a[2] * b[0];
        float a3b2 = a[2] * b[1];

        return {a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1};
    }

    mat3 transpose(const mat3 &m)
    {
        return {
            m.column(0),
            m.column(1),
            m.column(2)};
    }

    vec3 operator*(const mat3 &m, const vec3 &v)
    {
        vec3 res{};
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                res[i] = m[i][j] * v[j];
            }
        }
        return res;
    }

    inline quaternion operator*(const quaternion &q1, const float s)
    {
        return {q1.x * s, q1.y * s, q1.z * s, q1.w * s};
    }

    inline quaternion operator/(const quaternion &q1, const float s)
    {
        float inv_s = 1.0f / s;
        return {q1.x * inv_s, q1.y * inv_s, q1.z * inv_s, q1.w * inv_s};
    }

    inline quaternion from_axis_angle(const vec3 &axis, const float angle)
    {
        return {sinf(0.5f * angle) * normalize(axis), cosf(0.5f * angle)};
    }

    inline float length_squared(quaternion v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    inline float length(quaternion v)
    {
        return sqrtf(length_squared(v));
    }

    inline quaternion normalize(quaternion v)
    {
        return v / length(v);
    }

    inline mat4 identity_mat4()
    {
        return mat4{
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}};
    };

    mat4 operator*(const mat4 &m1, const mat4 &m2)
    {
        const vec4 &src_a0 = m1.rows[0];
        const vec4 &src_a1 = m1.rows[1];
        const vec4 &src_a2 = m1.rows[2];
        const vec4 &src_a3 = m1.rows[3];

        const vec4 &src_b0 = m2.rows[0];
        const vec4 &src_b1 = m2.rows[1];
        const vec4 &src_b2 = m2.rows[2];
        const vec4 &src_b3 = m2.rows[3];

        return mat4{
            src_a0 * src_b0[0] + src_a1 * src_b0[1] + src_a2 * src_b0[2] + src_a3 * src_b0[3],
            src_a0 * src_b1[0] + src_a1 * src_b1[1] + src_a2 * src_b1[2] + src_a3 * src_b1[3],
            src_a0 * src_b2[0] + src_a1 * src_b2[1] + src_a2 * src_b2[2] + src_a3 * src_b2[3],
            src_a0 * src_b3[0] + src_a1 * src_b3[1] + src_a2 * src_b3[2] + src_a3 * src_b3[3]};
    }

    mat4 &operator/=(mat4 &m, const float s)
    {
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                m.data[i][j] *= s;
            }
        }
        return m;
    }

    vec4 operator*(const mat4 &m, const vec4 &v)
    {
        vec4 res{};
        for (size_t i = 0; i < 4; i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                res[i] += m[i][j] * v[j];
            }
        }
        return res;
    }

    vec3 get_right(const mat4 &m)
    {
        return {m[0][0], m[0][1], m[0][2]};
    }

    vec3 get_up(const mat4 &m)
    {
        return {m[1][0], m[1][1], m[1][2]};
    }

    vec3 get_dir(const mat4 &m)
    {
        return {m[2][0], m[2][1], m[2][2]};
    }

    mat4 look_at(vec3 pos, vec3 target, vec3 world_up)
    {
        vec3 dir = normalize(pos - target);
        vec3 right = normalize(cross(world_up, dir));
        vec3 up = cross(dir, right);

        return {
            vec4{right, -dot(right, pos)},
            vec4{up, -dot(up, pos)},
            vec4{dir, -dot(dir, pos)},
            vec4{0.0f, 0.0f, 0.0f, 1.0f},
        };
    }

    mat4 invert(const mat4 &m)
    {
        // Code based on inversion code in GLM
        float s00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        float s01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        float s02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        float s03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        float s04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        float s05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        float s06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
        float s07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
        float s08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
        float s09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
        float s10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
        float s11 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
        float s12 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
        float s13 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
        float s14 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
        float s15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
        float s16 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
        float s17 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

        mat4 inv{};
        inv[0][0] = +(m[1][1] * s00 - m[1][2] * s01 + m[1][3] * s02);
        inv[0][1] = -(m[1][0] * s00 - m[1][2] * s03 + m[1][3] * s04);
        inv[0][2] = +(m[1][0] * s01 - m[1][1] * s03 + m[1][3] * s05);
        inv[0][3] = -(m[1][0] * s02 - m[1][1] * s04 + m[1][2] * s05);

        inv[1][0] = +(m[0][1] * s00 - m[0][2] * s01 + m[0][3] * s02);
        inv[1][1] = -(m[0][0] * s00 - m[0][2] * s03 + m[0][3] * s04);
        inv[1][2] = +(m[0][0] * s01 - m[0][1] * s03 + m[0][3] * s05);
        inv[1][3] = -(m[0][0] * s02 - m[0][1] * s04 + m[0][2] * s05);

        inv[2][0] = +(m[0][1] * s06 - m[0][2] * s07 + m[0][3] * s08);
        inv[2][1] = -(m[0][0] * s06 - m[0][2] * s09 + m[0][3] * s10);
        inv[2][2] = +(m[0][0] * s07 - m[0][1] * s09 + m[0][3] * s11);
        inv[2][3] = -(m[0][0] * s08 - m[0][1] * s10 + m[0][2] * s11);

        inv[3][0] = +(m[0][1] * s12 - m[0][2] * s13 + m[0][3] * s14);
        inv[3][1] = -(m[0][0] * s12 - m[0][2] * s15 + m[0][3] * s16);
        inv[3][2] = +(m[0][0] * s13 - m[0][1] * s15 + m[0][3] * s17);
        inv[3][3] = -(m[0][0] * s14 - m[0][1] * s16 + m[0][2] * s17);

        float det = +m[0][0] * inv[0][0] + m[0][1] * inv[0][1] + m[0][2] * inv[0][2] + m[0][3] * inv[0][3];

        inv /= det;
        return inv;
    }

    mat4 transpose(const mat4 &m)
    {
        return {
            m.column(0),
            m.column(1),
            m.column(2),
            m.column(3)};
    }
    mat3 to_mat3(const mat4 &m)
    {
        return {
            vec3{m[0][0], m[0][1], m[0][2]},
            vec3{m[1][0], m[1][1], m[1][2]},
            vec3{m[2][0], m[2][1], m[2][2]},
        };
    }

    inline void rotate(mat4 &mat, const quaternion &q)
    {
        mat = quat_to_mat(q) * mat;
    }

    inline void set_translation(mat4 &mat, const vec3 &translation)
    {
        mat[0][3] = translation.x;
        mat[1][3] = translation.y;
        mat[2][3] = translation.z;
    };

    mat4 perspective_projection(const float aspect_ratio, const float fov, const float z_near, const float z_far)
    {
        const float h = 1.0f / tanf(0.5f * fov);
        const float w = h / aspect_ratio;

        return mat4{
            {w, 0.0f, 0.0f, 0.0f},
            {0.0f, h, 0.0f, 0.0f},
            {0.0f, 0.0f, -(z_far) / (z_near - z_far) - 1, -(z_near * z_far) / (z_near - z_far)},
            {0.0f, 0.0f, -1.0f, 0.0f},
        };
    };
} // namespace zec
#endif // ZEC_MATH_IMPLEMENTATION
