#include <string.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "frame_struct.h"

frame_t *handle_process(std::string s) {
  static std::vector<uint8_t> vecChar;
  static const uint8_t sflag_l = FRAME_BEGIN_FLAG & 0xff;
  static const uint8_t sflag_h = (FRAME_BEGIN_FLAG >> 8) & 0xff;
  static const uint8_t eflag = FRAME_END_FLAG & 0xff;

  uint32_t frame_payload_len = 0;
  frame_t *pf = NULL;
  std::vector<uint8_t>::iterator it;
  
  vecChar.insert(vecChar.end(), s.cbegin(), s.cend());

  if (vecChar.size() < 2) {
    std::cout << "data is not enough!" << std::endl;
    return NULL;
  }

while(true){
  it = vecChar.begin();
  do {
    /* find sflag_h from [1:] first and next in [it+1:] */
    it = find(it + 1, vecChar.end(), sflag_h);
    /* sflag_h not found */
    if (it == vecChar.end()) {
      /* keep last element which may be sflag_l */
      std::vector<uint8_t>(vecChar.end() - 1, vecChar.end()).swap(vecChar);
      std::cout << "frame head not found! wait more data." << std::endl;
      return NULL;
    }
    /* sflag_h found, *(it-1) always valid */
  } while (*(it - 1) != sflag_l);
  /* we got *it==sflag_h and *(it-1)==sflag_l */

  if (it - 1 != vecChar.begin()) {
    std::vector<uint8_t>(it - 1, vecChar.end()).swap(vecChar);
    std::cout << "found! frame move to first!" << std::endl;
  }

  if (vecChar.size() < sizeof(frame_t)) {
    std::cout << "frame head data is not enough now! wait more data." << std::endl;
    return NULL;
  }

  pf = (frame_t *)&vecChar[0];
  frame_payload_len = pf->frame_head.frame_data_len - FRAME_HEAD_DATA_SIZE;

  /* max frame payload size */
  if (frame_payload_len > 100 * 100) {
    std::vector<uint8_t>(vecChar.begin()+1, vecChar.end()).swap(vecChar);
    std::cout<<"frame payload length: "<< frame_payload_len <<" > 10000"<<std::endl;
    continue;
  }

  if (vecChar.begin() + FRAME_HEAD_SIZE + frame_payload_len +
          FRAME_CHECKSUM_SIZE + FRAME_END_SIZE + 1 >
      vecChar.end()) {
    //std::cout << "expected frame payload length: " << frame_payload_len << std::endl;
    //std::cout << "frame payload data is not enough now! wait more data." << std::endl;
    return NULL;
  }

  {
    uint8_t check_sum = 0;
    for (uint32_t i = 0; i < FRAME_HEAD_SIZE + frame_payload_len; i++) {
      check_sum += ((uint8_t *)pf)[i];
    }
    if (check_sum != ((uint8_t *)pf)[FRAME_HEAD_SIZE + frame_payload_len] ||
        eflag != ((uint8_t *)pf)[FRAME_HEAD_SIZE + frame_payload_len +
                                 FRAME_CHECKSUM_SIZE]) {
      std::cout << "frame checksum or tail invalid! one more time." << std::endl;
      std::vector<uint8_t>(it, vecChar.end()).swap(vecChar);
      continue;
    }
  }

  pf = (frame_t *)malloc(sizeof(frame_t) + frame_payload_len);
  memcpy(pf, &vecChar[0], sizeof(frame_t) + frame_payload_len);

  std::vector<uint8_t>(it + FRAME_HEAD_SIZE + frame_payload_len +
                           FRAME_CHECKSUM_SIZE + FRAME_END_SIZE - 1,
                       vecChar.end())
      .swap(vecChar);
  return pf;
}
}
