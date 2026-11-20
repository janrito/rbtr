#pragma once

class Base {
 public:
  bool check() { return true; }
  bool save() { return check(); }
};
