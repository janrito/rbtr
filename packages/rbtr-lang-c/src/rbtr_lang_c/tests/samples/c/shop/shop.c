#include "shop.h"

static int valid(int n) {
    return n > 0;
}

int shop_total(int n) {
    return valid(n) ? n : 0;
}
