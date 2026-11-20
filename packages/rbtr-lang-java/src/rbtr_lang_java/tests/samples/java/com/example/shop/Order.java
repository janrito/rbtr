package com.example.shop;

import com.example.format.Formatter;

/** A subclass: `Model` is in its package, so it needs no import. */
public class Order extends Model {
    @Override
    public boolean save() {
        super.save();
        return check() && this.check();
    }

    static Order build() {
        return new Order();
    }
}
