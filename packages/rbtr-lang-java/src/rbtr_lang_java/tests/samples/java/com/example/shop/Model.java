package com.example.shop;

/** A base whose members a subclass reaches without naming them. */
public class Model {
    public boolean save() {
        return check();
    }

    public boolean check() {
        return true;
    }
}
