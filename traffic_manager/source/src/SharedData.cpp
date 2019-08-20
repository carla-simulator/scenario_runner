// Definition of class members

#include "SharedData.h"

namespace traffic_manager {

  SharedData::SharedData() {}
  SharedData::~SharedData() {}

  void SharedData::destroy() {
    for (auto actor: registered_actors) {
      actor->Destroy();
    }
  }

}
