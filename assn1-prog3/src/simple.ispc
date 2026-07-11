export void simple_add(uniform float a[], uniform float b[],
                       uniform float out[], uniform int n) {
    foreach (i = 0 ... n) {
        out[i] = a[i] + b[i];
    }
}
